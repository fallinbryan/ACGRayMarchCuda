# ACGRayMarchCuda
**Assignment 02: Advanced Ray Tracing in the Context of Advanced Computer Graphics, Masters Programme**

ACGRayMarchCuda is engineered as a component of the coursework for Advanced Computer Graphics, leveraging C++ and CUDA for parallelized ray tracing computations. This implementation focuses on the ray marching algorithm to address the computational challenges inherent in rendering detailed graphics scenes.

## Features

### Stratified Jittering for Ray Dispersion
Implements a stratified jittering mechanism to distribute multiple rays per pixel, thereby minimizing aliasing and enhancing the accuracy of lighting models.

### Parallel Computation on GPUs via CUDA
Exploits CUDA's parallel processing capabilities to accelerate the computations required for ray marching, significantly reducing processing times.

### Spatial Partitioning with Octree
Utilizes an Octree for spatial partitioning, enabling efficient space culling that enhances rendering performance for complex geometries.
![Octree Optimization](https://github.com/fallinbryan/ACGRayMarchCuda/assets/8240578/ff09b572-ddf4-465e-a858-5df9406e3702)

### Rendering of Soft Shadows
Achieves the rendering of soft shadows, thereby augmenting the realism of scenes without necessitating area lights.
![Soft Shadows](https://github.com/fallinbryan/ACGRayMarchCuda/assets/8240578/d8900051-fc10-4c88-9036-16b986e975e3)

### Depth of Field Effects
Facilitates depth of field effects, supporting dynamic focus in visual composition for cinematic rendering.
![Depth of Field](https://github.com/fallinbryan/ACGRayMarchCuda/assets/8240578/2255bd7d-ebc5-4736-857a-7ee9d4ab1e31)

## Limitations

- **Material Models**: Currently restricted to Blinn-Phong shading, limiting the diversity of material appearances.
- **Light Sources**: Solely supports directional lighting, constraining the variability in lighting scenarios.
- **Light Interaction**: Confined to a single shadow bounce, which may not accurately represent complex lighting environments.

## Scene File Format Specification

The project adopts a proprietary scene file format for the specification of scene elements. Below is an example manifesting the declaration syntax for a spherical object:

```
OBJECT
{
  TYPE SPHERE
  POSITION 3.5 -2.0 -1.5
  COLOR 0 0 255 255
}
```

This concise format facilitates the explicit definition of geometric primitives and their attributes within scene compositions.
