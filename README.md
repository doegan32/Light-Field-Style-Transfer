# Light Field Style Transfer With Local Angular Consistency
Implementation of Light Field Style Transfer with Local Angular Consistency


## Light field conventions

## Metrics
We also provide MATLAB scripts for the evalation metrics used in the paper - the LFEC metric [2] and our proposed LFAC metric [1]. 

### Light Field Epipolar Consistency (LFEC) metric
The script for the LFEC is in LightFieldEpipolarConsistency.m. It takes as input the light field (as an array of dimension (s,t,u,v,c)) to be evaluated as well as the disparity map for the centre view of the original (i.e. non-edited) light field (as an array of dimension (u,v)). The disparity map can be the ground truth map if avaliable or it can be estimated by some other means. 

### Light Field Angular Consistency (LFAC) metric
The script for the LFEC is in LightFieldAngularConsistency.m. It takes as input the light field (as an array of dimension (s,t,u,v,c)) to be evaluated as well as the disparity map for <\b>each view<\b> of the original (i.e. non-edited) light field (as an array of dimension (u,v)). The disparity map can be the ground truth map if avaliable or it can be estimated by some other means. 




## References
[2] P. David, M. L. Pendu, and C. Guillemot, “Angularly consistent light field video interpolation,” in Proc. IEEE ICME, 2020, pp. 1–6
