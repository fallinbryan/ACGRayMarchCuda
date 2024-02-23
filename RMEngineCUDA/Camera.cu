#include "Camera.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>



//__device__ glm::vec2 Camera::getNormalizedDeviceCoordinates(int x, int y, int width, int height) const {
//  return glm::vec2(
//    (2.0f * x) / width - 1.0f,
//    1.0f - (2.0f * y) / height
//  );
//}



__device__ void Camera::normalizeCoordinates(glm::vec2& pixel, float width, float height) const {
  float aspectRatio = (float)width / (float)height;

  pixel.x = (2.0f * pixel.x / width - 1.0f) * aspectRatio;
  pixel.y = 1.0f - (2.0f * pixel.y) / height;
}

__device__ void Camera::scaleCoordinates(glm::vec2& pixel) {
  float radians = fov * PI / 180.0f;
  float scale = tan(radians / 2.0f);
  pixel.x *= scale;
  pixel.y *= scale;
}

__device__ void Camera::toWorldSpace(glm::vec3& rayDirection) {
  glm::mat4 viewMatrix = glm::lookAt(position, lookAt, up);
  glm::mat4 invView = glm::inverse(viewMatrix);
  glm::vec4 newDirection = invView * glm::vec4(rayDirection, 0.0f);
  rayDirection = glm::vec3(newDirection);
  glm::normalize(rayDirection);
}

__device__ bool Camera::wtf() const {
  return true;
}