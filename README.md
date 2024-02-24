# ACGRayMarchCuda
**Assignment: 02 Advanced Ray Tracing - Part of the Advanced Computer Graphics (ACG) Course for Masters Degree**

ACGRayMarchCuda is a ray marching engine developed for Assignment 02 in the Advanced Computer Graphics course. This project demonstrates the application of C++ and CUDA in achieving parallel processing for ray tracing, incorporating several optimization techniques and rendering features to meet the course's requirements.

## Features

### Shooting Multiple Rays per Pixel via a Stratified Jittering Approach
Implements a stratified jittering approach for shooting multiple rays per pixel, enhancing image quality and realism by reducing aliasing and providing more accurate lighting and shadow effects.


### Parallel Processing on GPU using CUDA
Utilizes NVIDIA CUDA technology for parallel processing on GPUs, enhancing the speed of ray marching calculations.

### Octree Optimization
Employs an Octree data structure for efficient space culling, improving the rendering speed of complex scenes.

![Octree Optimization](https://github.com/fallinbryan/ACGRayMarchCuda/assets/8240578/ff09b572-ddf4-465e-a858-5df9406e3702)

### Soft Shadows
Features soft shadow rendering to increase realism in scenes, achieved without the use of area lights.

![Soft Shadows](https://github.com/fallinbryan/ACGRayMarchCuda/assets/8240578/d8900051-fc10-4c88-9036-16b986e975e3)

### Depth of Field
Supports depth of field effects for cinematic rendering and focus effects in visual compositions.

![Depth of Field](https://github.com/fallinbryan/ACGRayMarchCuda/assets/8240578/2255bd7d-ebc5-4736-857a-7ee9d4ab1e31)
