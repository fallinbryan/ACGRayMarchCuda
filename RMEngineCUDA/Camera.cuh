#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#define PI 3.14159265359f

struct Camera
{
  glm::vec3 position;
  glm::vec3 up;
  glm::vec3 lookAt;
  float fov;
  float aspect;
  float near;
  float far;
  float focalLength;
  float aperture;



  __device__ void normalizeCoordinates(glm::vec2& pixel, float width, float height) const;
  __device__ void scaleCoordinates(glm::vec2& pixel);
  __device__ void toWorldSpace(glm::vec3& direction);


  __device__ bool wtf() const;
};