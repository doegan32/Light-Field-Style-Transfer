# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:09:51 2020

@author: donal
"""

import argparse
import os
import struct
import time

from PIL import Image
import cv2 as cv
import numpy as np

import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

def parse_args():
    # desc = "PyTorch implementation of paper nueral style transfer for light fields"
    # parser = argparse.ArgumentParser(desc)
    parser = argparse.ArgumentParser()

    #light field arguments
    parser.add_argument('--lf_input_dir', type = str, default = './lf_input', help='place light field to be stylized in this directory')
    parser.add_argument('--lf_output_dir', type = str, default = './lf_output', help='stylized light field will be saved to this directory')
    parser.add_argument('--content_img_frmt', type=str, default='viewpoint_{}_{}.png', help='Filename format of the input sub aperture images')
    parser.add_argument('--st_size', type = int, default = 9, help='angular dimensions of light field')
    parser.add_argument('--max_size', type=int, default=512, help='Maximum width or height of the input images')
    #style target
    parser.add_argument('--style_dir', type = str, default = './styles', help='save your style image here')
    parser.add_argument('--style_img', type = str, required = True, help='name of style image to be used')
    # optical flow arguments 
    parser.add_argument('--optic_flow_dir', type=str, default='./of_output', help='Relative or absolute directory path to optic flow inputs.')
    parser.add_argument('--optic_flow_frmt_bw', type=str, default='backward_{}_{}_{}_{}.flo', help='Filename format of the backward optical flow files.')
    parser.add_argument('--optic_flow_frmt_fw', type=str, default='forward_{}_{}_{}_{}.flo', help='Filename format of the forward optical flow files')
    parser.add_argument('--consistency_weights_frmt', type=str, default='reliable_{}_{}_{}_{}.txt', help='Filename format of the optical flow consistency files.')
    # optimization options
    parser.add_argument('--content_weight', type=float, default=5e0, help='Weight for the content loss function.')
    parser.add_argument('--style_weight', type=float, default= 1e4, help='Weight for the style loss function.')
    parser.add_argument('--angular_weight', type=float, default=2e5, help='Weight for the temporal loss function.')
    parser.add_argument('--iters_first', type = int, default = 1000)
    parser.add_argument('--iters', type = int, default = 900)
    parser.add_argument('--device', type = str, default = 'cuda', choices = ['cpu', 'cuda'])
    # other
    parser.add_argument('--model_dir', type=str, default='./models', help='directory containing pre-trained VGG19 model.')
    parser.add_argument('--model_weights', type=str, default='vgg_conv.pth', help='Weights and biases of the VGG-19 network.')
    parser.add_argument('--content_layers', nargs='+', type=str, default=['r42'], help='VGG19 layers used for the content loss.')
    parser.add_argument('--style_layers', nargs='+', type=str, default=['r11', 'r21', 'r31', 'r41', 'r51'], help='VGG19 layers used for the style loss.')
    parser.add_argument('--counter', type = int, default = 1)
    parser.add_argument('--warp_img_dir', type=str, default='./warps', help='Directory path to the warped images.')
    return parser.parse_args()

def read_flow_file(path):
    with open(path, 'rb') as f:
        # 4 bytes header
        header = struct.unpack('4s', f.read(4))[0]
        # 4 bytes width, height    
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]   
        flow = np.ndarray((2, h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                flow[0,y,x] = struct.unpack('f', f.read(4))[0]
                flow[1,y,x] = struct.unpack('f', f.read(4))[0]
    return flow

def read_weights_file(path):
    lines = open(path).readlines()
    header = list(map(int, lines[0].split(' ')))
    w = header[0]
    h = header[1]
    vals = np.zeros((h, w), dtype=np.float32)
    for i in range(1, len(lines)):
        line = lines[i].rstrip().split(' ')
        vals[i-1] = np.array(list(map(np.float32, line)))
        vals[i-1] = list(map(lambda x: 0. if x < 255. else 1., vals[i-1]))
    
    # expand to 3 channels
    weights = np.dstack([vals.astype(np.float32)] * 3)
    weights = transforms.ToTensor()(weights).unsqueeze(0)
    return weights

def get_warped(vp1, vp2):
    """ warps viewpoint vp2 onto viewpoint vp1 """

    s1, t1 = vp1
    s2, t2 = vp2

    # read in frame to be warped - need to use openCV for this to get right format for cv.remap()?
    fn = args.content_img_frmt.format(str(s2).zfill(2), str(t2).zfill(2))
    path = os.path.join(args.lf_output_dir, fn)
    prev_img = cv.imread(path, cv.IMREAD_COLOR)
    
    # read in flow file: backward_current frame_previous frame
    fn = args.optic_flow_frmt_bw.format(str(s1), str(t1), str(s2), str(t2))
    path = os.path.join(args.optic_flow_dir, fn)
    flow = read_flow_file(path)
    
    # construct flow map from flow file
    _, h, w = flow.shape
    flow_map = np.zeros(flow.shape, dtype=np.float32)
    for y in range(h):
        flow_map[1,y,:] = float(y) + flow[1,y,:] #* (s1 != s2) # only adds in vertical if viewpoints in different rows
    for x in range(w):
        flow_map[0,:,x] = float(x) + flow[0,:,x] #* (t1 != t2) # only adds in horizontal if viewpoints in different columns
    
    # remap pixels to optical flow
    warped_img = cv.remap(prev_img, flow_map[0], flow_map[1], interpolation=cv.INTER_CUBIC, borderMode=cv.BORDER_TRANSPARENT)
    warped_img = cv.cvtColor(warped_img, cv.COLOR_BGR2RGB) #numpy array with dtype np.uint8
    #warped_img = preprocess(Image.fromarray(warped_img))
    return warped_img

def get_consistency_weights(vp1, vp2):
 
    s1, t1 = vp1
    s2, t2 = vp2
    
    #forward_fn = args.consistency_weights_frmt.format(str(s2), str(t2), str(s1), str(t1))
    backward_fn = args.consistency_weights_frmt.format(str(s1), str(t1), str(s2), str(t2))
    #forward_path = os.path.join(args.optic_flow_dir, forward_fn)
    backward_path = os.path.join(args.optic_flow_dir, backward_fn)
    #forward_weights = read_weights_file(forward_path)
    backward_weights = read_weights_file(backward_path)
    return backward_weights.to(args.device)#forward_weights.to(args.device) 

# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

def preprocess(image):
    img_size = (512, 512)
    prep = transforms.Compose([transforms.Resize(img_size), #args.max_size
                               transforms.ToTensor(), # converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
                               transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                    std=[1,1,1]),
                               transforms.Lambda(lambda x: x.mul_(255)),
                              ])
    image_torch = prep(image).unsqueeze(0).to(args.device)
    return image_torch

def postprocess(image):
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
    postpb = transforms.Compose([transforms.ToPILImage()]) 
    t = postpa(image)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img

# loss functions
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

class angLoss(nn.Module):
    def __init__(self, c):
        super(angLoss, self).__init__()
        self.c = c
    
    def forward(self, input, target):
        out = nn.MSELoss()(torch.mul(input,self.c),torch.mul(target, self.c))
        return out    

def stylize(style_img, content_indices, other_indices = []):
    """
        content_indices is a tuple
        other_indices is a list of tuples of viewpoints to be used for loss function and initialization
    """
    tick = time.time()
    s,t = content_indices
    print("Starting stylizing viewpoint ({:02d}, {:02d}) ({:d})".format(s, t, args.counter))
    report = open(os.path.join(args.lf_output_dir,"results.txt"), "a")
    report.writelines("\n\n\tStylizing viewpoint : ({:02d}, {:02d}) ({:d})".format(s, t, args.counter))

    
    if args.device == "cuda":
        torch.cuda.empty_cache()
    
    
    # get content image
    path = os.path.join(args.lf_input_dir, args.content_img_frmt.format(str(s).zfill(2), str(t).zfill(2)))
    content_img = preprocess(Image.open(path))
    _,c,h,w = content_img.shape
        
    # get init image
    if s == int(args.st_size/2) and t == int(args.st_size/2):
        init_img = content_img.data.clone()
        consistency_weights = [] # better way without this?
        angular_vps = [] # just added in for ease later on
        neighbour_weights = []
    else:
        neighbours_warped = np.zeros((len(other_indices), h, w, c), dtype = np.uint8)
        neighbour_weights = []
        consistency_weights = []
        angular_vps = [] #just added in for ease later on         
        for i, vp in enumerate(other_indices):
            s2, t2 = vp
            neighbours_warped[i] = get_warped(content_indices, vp)
            consistency_weights.append(get_consistency_weights(content_indices, vp))
            angular_vps.append("angular{:d}_{:d}".format(s2,t2))
            if s2 == s or t2 == t:
                neighbour_weights.append(1.0) 
            else:
                neighbour_weights.append(0.71)
        init_img = Image.fromarray(np.average(neighbours_warped, axis = 0, weights = neighbour_weights).astype(np.uint8))
        
        plt.figure()
        plt.imshow(init_img)
        plt.show()
        
        init_img = preprocess(init_img)
    init_img.to(args.device)
    init_img.requires_grad_()

    # save warp for now to examine
    initial = postprocess(init_img.data[0].cpu().squeeze())
    fn = args.content_img_frmt.format(str(s).zfill(2), str(t).zfill(2))
    path = os.path.join(args.warp_img_dir, fn)
    initial.save(path, "PNG")

    # construct loss functions
    style_loss_functions = [GramMSELoss().to(args.device)] * len(args.style_layers)
    content_loss_functions = [nn.MSELoss().to(args.device)] * len(args.content_layers)
    angular_loss_functions = [angLoss(c).to(args.device) for c in consistency_weights] # need to add in weights (as per Martin's slides) or really should add in below in weights section?
    loss_functions = style_loss_functions + content_loss_functions + angular_loss_functions 
    

    if s == int(args.st_size / 2) and t == int(args.st_size / 2):
        max_iters = args.iters_first
    else:
        max_iters = args.iters

    # weights - need to edit to incorporate correct weighting for each sub-component of loss functions
    style_weights = [(args.style_weight / len(args.style_layers)) / n**2 for n in [64, 128, 256, 512, 512]] # does the 1/n^2 that Gatys has not come from the MSE loss? each matrix has n^2 entries and so the MSE divides by n^2???
    content_weights = [args.content_weight for layer in args.content_layers]
    angular_weights = [args.angular_weight * (weight / sum(neighbour_weights)) for weight in neighbour_weights]
    weights = style_weights + content_weights + angular_weights
        
    # get network
    vgg = VGG()
    vgg.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_weights)))
    for param in vgg.parameters():
        param.requires_grad = False #means gradients don't need to be computed for tensor. 
    vgg.to(args.device)

    
    # prepare targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_img, args.style_layers)]
    content_targets = [A.detach() for A in vgg(content_img, args.content_layers)]
    angular_targets = [preprocess(Image.fromarray(neighbours_warped[i])) for i in range(len(other_indices))]
    targets = style_targets + content_targets + angular_targets
    
    # prepare style and content layers
    loss_layers = args.style_layers + args.content_layers#style and content 
    all_losses = args.style_layers + args.content_layers+ angular_vps #just lazy here to make below print sttement easier
    
    # plt.figure()
    # plt.imshow(postprocess(style_img.data[0].cpu().squeeze()))
    # plt.show()
    # plt.figure()
    # plt.imshow(postprocess(content_img.data[0].cpu().squeeze()))
    # plt.show()
    # plt.figure()
    # plt.imshow(postprocess(init_img.data[0].cpu().squeeze()))
    # plt.show()
    
    
    # optimization
    optimizer =  optim.LBFGS([init_img])
    n_iter = [0]
    while n_iter[0] <= max_iters:

        def closure():
            optimizer.zero_grad()
            out = vgg(init_img, loss_layers)
            out += [init_img] * len(other_indices)
            

            
            layer_losses = [weights[a] * loss_functions[a](A, targets[a]) for a, A in enumerate(out)]
            loss = sum(layer_losses)
            
            loss.backward()
            
            n_iter[0] += 1
            if n_iter[0] % 50 == 0: #make argument and fix up formatting
                print("Iteration: {:d}, loss: {:.4f}".format(n_iter[0], loss))
                print([all_losses[li] + ': ' +  str(l.item()) for li,l in enumerate(layer_losses)])
                report.writelines("\n\t\tIteration: {:d}, loss: {:0.4f}".format(n_iter[0], loss.item()))
                report.writelines(["\n\t\t\t" + all_losses[li] + ': ' +  str(l.item()) for li,l in enumerate(layer_losses)])
            
            return loss
        
        optimizer.step(closure)


    # plt.figure()
    # plt.imshow(postprocess(style_img.data[0].cpu().squeeze()))
    # plt.show()
    # plt.figure()
    # plt.imshow(postprocess(content_img.data[0].cpu().squeeze()))
    # plt.show()
    # plt.figure()
    # plt.imshow(postprocess(init_img.data[0].cpu().squeeze()))
    # plt.show()
    
    out_img = postprocess(init_img.data[0].cpu().squeeze())
    
    # plt.figure()
    # plt.imshow(out_img)
    # plt.show()
    
    
    fn = args.content_img_frmt.format(str(s).zfill(2), str(t).zfill(2))
    path = os.path.join(args.lf_output_dir, fn)
    out_img.save(path, "PNG")
    args.counter += 1
    tock = time.time()
    print("Time taken was {:.4f} seconds.".format(tock - tick))
    report.writelines("\n\t\tTime taken: {:.4f}".format(tock-tick))     
    report.close()

def main():
    
    tick = time.time() # set timer going
    
    global args
    args = parse_args()
    
    report = open(os.path.join(args.lf_output_dir,"results.txt"), "a")
    report.writelines("\nContent weight: {:f}".format(args.content_weight))
    report.writelines("\nStyle weight: {:f}".format(args.style_weight))
    report.writelines("\nAngular weight: {:f}".format(args.angular_weight))
    report.writelines("\n\nStyle: {:s}".format(args.style_img))
    report.writelines("\n\nIterations centre  frame: {:d}".format(args.iters_first))
    report.writelines("\n\nIterations other  frames: {:d}".format(args.iters))
    report.close()
    
    if args.device == "cuda":
        torch.cuda.empty_cache()
    
    m = args.st_size
    c = int(m/2)
    print("light-field has {:d}x{:d} sub-aperture images".format(m,m))

    #get style image
    path = os.path.join(args.style_dir, args.style_img)
    style_img = preprocess(Image.open(path))

    counter = 0 # keep track of number of viewpoints stylized. Can delete
    
    # step 0 - stylize central viewpoint
    stylize(style_img, (c, c))
    counter += 1
    
    # step 1 - stylize all viewpoints around central image
    # 4 midpoints
    stylize(style_img, (c-1, c), [(c, c)])
    stylize(style_img, (c, c+1), [(c, c)])
    stylize(style_img, (c+1, c), [(c, c)])
    stylize(style_img, (c, c-1), [(c, c)])
    # 4 corners
    stylize(style_img, (c-1, c-1), [(c, c), (c-1,c), (c, c-1)])
    stylize(style_img, (c-1, c+1), [(c, c), (c-1, c), (c,c+1)])
    stylize(style_img, (c+1, c+1), [(c, c), (c, c+1), (c+1, c)])
    stylize(style_img, (c+1, c-1), [(c, c), (c+1, c), (c, c-1)])
    counter += 8


    #step n - transfer style from all closest images stylized at step n-1
    for i in range(2, c+1):
        print("starting circle {:d}".format(i+1))
        
        #middle of sides - 3 warps
        stylize(style_img, (c-i, c), [(c+1-i, c-1), (c+1-i, c), (c+1-i, c+1)])
        stylize(style_img, (c, c+i), [(c-1, c-1+i), (c, c-1+i), (c+1, c-1+i)]) 
        stylize(style_img, (c+i, c), [(c-1+i, c+1), (c-1+i, c), (c-1+i, c-1)])
        stylize(style_img, (c, c-i), [(c+1, c+1-i), (c, c+1-i), (c-1, c+1-i)])
        counter += 4
        
        # rest of sides excluding corners and adjacent to corners - 3 warps
        for j in range(1,i-1):
            stylize(style_img, (c-i, c-j), [(c+1-i, c-1-j), (c+1-i, c-j), (c+1-i, c+1-j), (c-i, c-j+1)])
            stylize(style_img, (c-i, c+j), [(c+1-i, c-1+j), (c+1-i, c+j), (c+1-i, c+1+j), (c-i, c+j-1)])
            stylize(style_img, (c-j, c+i), [(c-1-j, c-1+i), (c-j, c-1+i), (c+1-j, c-1+i), (c-j+1, c+i)])
            stylize(style_img, (c+j, c+i), [(c-1+j, c-1+i), (c+j, c-1+i), (c+1+j, c-1+i), (c+j-1, c+i)])
            stylize(style_img, (c+i, c+j), [(c-1+i, c+1+j), (c-1+i, c+j), (c-1+i, c-1+j), (c+i, c+j-1)])
            stylize(style_img, (c+i, c-j), [(c-1+i, c+1-j), (c-1+i, c-j), (c-1+i, c-1-j), (c+i, c-j+1)])
            stylize(style_img, (c+j, c-i), [(c+1+j, c+1-i), (c+j, c+1-i), (c-1+j, c+1-i), (c+j-1, c-i)])
            stylize(style_img, (c-j, c-i), [(c+1-j, c+1-i), (c-j, c+1-i), (c-1-j, c+1-i), (c-j+1, c-i)])
            counter += 8
    
        # adjacent to corners - 2 warps
        stylize(style_img, (c-i, c-i+1), [(c-i+1, c-i+1), (c-i+1, c-i+2), (c-i, c-i+2)])
        stylize(style_img, (c-i, c+i-1), [(c-i+1, c+i-2), (c-i+1, c+i-1), (c-i, c+i-2)])
        stylize(style_img, (c-i+1, c+i), [(c-i+1, c+i-1), (c-i+2, c+i-1), (c-i+2, c+i)])
        stylize(style_img, (c+i-1, c+i), [(c+i-2, c+i-1), (c+i-1, c+i-1), (c+i-2, c+i)])
        stylize(style_img, (c+i, c+i-1), [(c+i-1, c+i-1), (c+i-1, c+i-2), (c+i, c+i-2)])
        stylize(style_img, (c+i, c-i+1), [(c+i-1, c-i+2), (c+i-1, c-i+1), (c+i, c-i+2)])
        stylize(style_img, (c+i-1, c-i), [(c+i-1, c-i+1), (c+i-2, c-i+1), (c+i-2, c-i)])
        stylize(style_img, (c-i+1, c-i), [(c-i+2, c-i+1), (c-i+1, c-i+1), (c-i+2, c-i)])
        counter += 8
        
        # 4 corners - one warp
        stylize(style_img, (c-i, c-i), [(c+1-i, c+1-i), (c-i+1, c-i), (c-i, c-i+1)])
        stylize(style_img, (c-i, c+i), [(c+1-i, c-1+i), (c-i, c+i-1), (c-i+1, c+i)])
        stylize(style_img, (c+i, c+i), [(c-1+i, c-1+i), (c+i-1, c+i), (c+i, c+i-1)])
        stylize(style_img, (c+i, c-i), [(c-1+i, c+1-i), (c+i, c-i+1), (c+i-1, c-i)])
        counter += 4
     
    tock = time.time()
    total_time = (tock - tick) / 3600.0
    print("Number of frames stylized was: {:d}.\n\
          Total time taken was {:0.4f} hours.".format(counter, total_time))
          
    report = open(os.path.join(args.lf_output_dir,"results.txt"), "a")
    report.writelines("\nNumber of frames stylized: {:d}".format(counter))
    report.writelines("\nTotal time taken: {:0.4f} hours\n".format(total_time))
    report.close()
                    
if __name__ == '__main__':
    main()