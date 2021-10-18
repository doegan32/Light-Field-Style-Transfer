# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:25:54 2020

@author: donal

Run this script to estimate disparity between the neighbouring light field viewpoints
and to also calculate the weights c_{s't'}^{s,t} defined in section 2.2.2 of the paper.
DeepFlow (http://lear.inrialpes.fr/src/deepflow/) is used for this.
The code is adapted from that found here https://github.com/manuelruder/artistic-videos 

"""
import argparse
import os
import time

def parse_args():
    
    parser = argparse.ArgumentParser()
    #light field arguments
    parser.add_argument('--lf_input_dir', type = str, default = './lf_input', help='directory in which input light field is saved')
    parser.add_argument('--content_img_frmt', type=str, default='view_%02d_%02d.png', help='Filename format of the input content frames.')
    parser.add_argument('--st_size', type = int, default = 9)
    # optical flow arguments 
    parser.add_argument('--optic_flow_dir', type=str, default='./of_output', help='disparity/optic flow estimates will be saved here')
    parser.add_argument('--optic_flow_frmt_bw', type=str, default='backward_{}_{}_{}_{}.flo', help='Filename format of the backward optical flow files.')
    parser.add_argument('--optic_flow_frmt_fw', type=str, default='forward_{}_{}_{}_{}.flo', help='Filename format of the forward optical flow files')
    parser.add_argument('--consistency_weights_frmt', type=str, default='reliable_{}_{}_{}_{}.txt', help='Filename format of the optical flow consistency files.')
    return parser.parse_args()

def optic_flow(vp, neighbours = []):
    
    s1, t1 = vp
    
    for vp2 in neighbours:
        s2, t2 = vp2
    
        # read in flow file: backward_current frame_previous frame
        fn = args.optic_flow_frmt_bw.format(str(s1), str(t1), str(s2), str(t2))
        path = os.path.join(args.optic_flow_dir, fn)
        if not os.path.exists(path):
            os.system("./make-opt-flow.sh " + args.lf_input_dir + "/" + args.content_img_frmt + " " + args.optic_flow_dir +" {:d} {:d} {:d} {:d}".format(s1, t1, s2, t2))

def main():
    tick = time.time() # set timer going
    
    global args
    args = parse_args()
    
    m = args.st_size
    c = int(m/2)
    print("light-field has {:d}x{:d} sub-aperture images".format(m,m))

    counter = 0 # keep track of number of viewpoints completed
   
    # step 0 - central viewpoint (doesn't do anything as set of neighbours is empty)
    optic_flow((c, c))
    counter += 1
    
    # step 1 - estimate disparity between central view and surrounding views
    # 4 midpoints
    optic_flow((c-1, c), [(c, c)])
    optic_flow((c, c+1), [(c, c)])
    optic_flow((c+1, c), [(c, c)])
    optic_flow((c, c-1), [(c, c)])
    # 4 corners
    optic_flow( (c-1, c-1), [(c, c), (c-1,c), (c, c-1)])
    optic_flow( (c-1, c+1), [(c, c), (c-1, c), (c,c+1)])
    optic_flow( (c+1, c+1), [(c, c), (c, c+1), (c+1, c)])
    optic_flow( (c+1, c-1), [(c, c), (c+1, c), (c, c-1)])
    counter += 8

    #step n 
    for i in range(2, c+1):
        print("starting circle {:d}".format(i+1))
        
        # middle of sides - 3 warps
        optic_flow((c-i, c), [(c+1-i, c-1), (c+1-i, c), (c+1-i, c+1)])
        optic_flow((c, c+i), [(c-1, c-1+i), (c, c-1+i), (c+1, c-1+i)]) 
        optic_flow((c+i, c), [(c-1+i, c+1), (c-1+i, c), (c-1+i, c-1)])
        optic_flow((c, c-i), [(c+1, c+1-i), (c, c+1-i), (c-1, c+1-i)])
        counter += 4
        
        # rest of sides excluding corners and adjacent to corners - 3 warps
        for j in range(1,i-1):
            optic_flow( (c-i, c-j), [(c+1-i, c-1-j), (c+1-i, c-j), (c+1-i, c+1-j), (c-i, c-j+1)])
            optic_flow( (c-i, c+j), [(c+1-i, c-1+j), (c+1-i, c+j), (c+1-i, c+1+j), (c-i, c+j-1)])
            optic_flow( (c-j, c+i), [(c-1-j, c-1+i), (c-j, c-1+i), (c+1-j, c-1+i), (c-j+1, c+i)])
            optic_flow( (c+j, c+i), [(c-1+j, c-1+i), (c+j, c-1+i), (c+1+j, c-1+i), (c+j-1, c+i)])
            optic_flow( (c+i, c+j), [(c-1+i, c+1+j), (c-1+i, c+j), (c-1+i, c-1+j), (c+i, c+j-1)])
            optic_flow( (c+i, c-j), [(c-1+i, c+1-j), (c-1+i, c-j), (c-1+i, c-1-j), (c+i, c-j+1)])
            optic_flow( (c+j, c-i), [(c+1+j, c+1-i), (c+j, c+1-i), (c-1+j, c+1-i), (c+j-1, c-i)])
            optic_flow( (c-j, c-i), [(c+1-j, c+1-i), (c-j, c+1-i), (c-1-j, c+1-i), (c-j+1, c-i)])
            counter += 8
    
        # adjacent to corners - 2 warps
        optic_flow( (c-i, c-i+1), [(c-i+1, c-i+1), (c-i+1, c-i+2), (c-i, c-i+2)])
        optic_flow( (c-i, c+i-1), [(c-i+1, c+i-2), (c-i+1, c+i-1), (c-i, c+i-2)])
        optic_flow( (c-i+1, c+i), [(c-i+1, c+i-1), (c-i+2, c+i-1), (c-i+2, c+i)])
        optic_flow( (c+i-1, c+i), [(c+i-2, c+i-1), (c+i-1, c+i-1), (c+i-2, c+i)])
        optic_flow( (c+i, c+i-1), [(c+i-1, c+i-1), (c+i-1, c+i-2), (c+i, c+i-2)])
        optic_flow( (c+i, c-i+1), [(c+i-1, c-i+2), (c+i-1, c-i+1), (c+i, c-i+2)])
        optic_flow( (c+i-1, c-i), [(c+i-1, c-i+1), (c+i-2, c-i+1), (c+i-2, c-i)])
        optic_flow( (c-i+1, c-i), [(c-i+2, c-i+1), (c-i+1, c-i+1), (c-i+2, c-i)])
        counter += 8
        
        # 4 corners - one warp
        optic_flow( (c-i, c-i), [(c+1-i, c+1-i), (c-i+1, c-i), (c-i, c-i+1)])
        optic_flow( (c-i, c+i), [(c+1-i, c-1+i), (c-i, c+i-1), (c-i+1, c+i)])
        optic_flow( (c+i, c+i), [(c-1+i, c-1+i), (c+i-1, c+i), (c+i, c+i-1)])
        optic_flow( (c+i, c-i), [(c-1+i, c+1-i), (c+i, c-i+1), (c+i-1, c-i)])
        counter += 4
     
    tock = time.time()
    total_time = (tock - tick) / 3600.0
    print("Total time taken was {:0.4f} hours.".format(total_time))
          
if __name__ == '__main__':
    main()