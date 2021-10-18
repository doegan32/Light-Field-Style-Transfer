# Light Field Style Transfer With Local Angular Consistency
This repository contains an implementation for the paper Light Field Style Transfer with Local Angular Consistency [1].

Additional information and sample results are available on the [project webpage](https://v-sense.scss.tcd.ie/research/neural-style-transfer-for-light-fields/)

## Light Field Style transfer
### Baselines

**Image style transfer**
Each sub-aperture image of the light field is independently stylized using the method described in [2]. For this, we use the code provided [here](https://github.com/leongatys/PytorchNeuralStyleTransfer). 

**Video style transfer**
A pseudovideo is createad from the light field sub-aperture images, for example, by scanning them in a snake like order from top to bottom or in a spiral order from the centre towards the edge. The video style transfer method described in [3] is then applied to this pseudovideo. An implementation of this method is provided [here](https://github.com/manuelruder/artistic-videos).

### Style transfer with local angular consistency
**Preparation**
- The disparity between neighbouring light field viewpoints and the weights c<sub>s',t'<sub><sup>s,t<sup> (as described in section 2.2.2 of the paper) must be calculated in advance. This can be done using wichever method you choose. We used [DeepFlow](http://lear.inrialpes.fr/src/deepflow/). Run the script . For this to work you will need to download DeepFlow and DeepMatching from their [website](http://lear.inrialpes.fr/src/deepflow/) and save the static binaries (deepmatching-static and deepflow2-static) in the main directory of this repository.

- The code requires the pretrained VGG19 network [5]. Download this [here](https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth) and save to a directory called "/models". 
- Saved your desired style image to a directory called "/styles". An example style image is provided with this repository.

The disparity maps for neighbouring light field views must be calculated. 


## Metrics
We also provide MATLAB scripts for the evalation metrics used in the paper - the LFEC metric [4] and our proposed LFAC metric [1]. Explanations of the metrics are contained in the relevant papers. 

### Light Field Epipolar Consistency (LFEC) metric
The script for the LFEC is in LightFieldEpipolarConsistency.m. It takes as input the light field to be evaluated (as an array of dimension (s,t,u,v,c)) as well as the disparity map for the **centre view** of the original (i.e. non-edited) light field (as an array of dimension (u,v)). The disparity map can be the ground truth map if avaliable or it can be estimated by some other means. 

### Light Field Angular Consistency (LFAC) metric
The script for the LFAC is in LightFieldAngularConsistency.m. It takes as input the light field to be evaluated (as an array of dimension (s,t,u,v,c)) as well as the disparity map for **each view** of the original (i.e. non-edited) light field (as an array of dimension (s,t,u,v)). The disparity map can be the ground truth map if avaliable or it can be estimated by some other means, for example, using ?. 




## References
[1] D. Egan, M. Alain and A. Smolic, "Light Field Style Transfer with Local Angular Consistency," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 2300-2304, doi: 10.1109/ICASSP39728.2021.9414689.

[2] L. A. Gatys, A. S. Ecker and M. Bethge, "Image style transfer using convolutional neural networks", Proc. IEEE CVPR, pp. 2414-2423, 2016.

[3] M. Ruder, A. Dosovitskiy and T. Brox, "Artistic style transfer for videos and spherical images", International Journal of Computer Vision, vol. 126, no. 11, pp. 1199-1219, 2018.

[4] P. David, M. L. Pendu, and C. Guillemot, “Angularly consistent light field video interpolation,” in Proc. IEEE ICME, 2020, pp. 1–6

[5] K. Simonyan and Andrew Zisserman, “Very deep convolutional networks for large-scale image recognition,” CoRR, vol. abs/1409.1556, 2015
