

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <random>

#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"




#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "GLUtil.h"
#include "Camera.cuh"
#include "RayMarching.cuh"
#include "cutils.cuh"
#include "Octree.h"



#define SCENE_WIDTH 1280
#define SCENE_HEIGHT 720

//#define SCENE_WIDTH 800
//#define SCENE_HEIGHT 600

#define ENABLE_ANTI_ALIASING true
#define RENDER_NORMALS false
#define DEBUG_VIZUALIZE_OCTREE false

#define WITH_DOF true
#define APERTURE 0.3f
#define FOCAL_DISTANCE 10.05f

#define ADD_RANDOM_SPHERES false
#define SQRT_SAMPLE_PER_PIXEL 8

#define RANDOM_MIN_RANGE 0.0f
#define RANDOM_MAX_RANGE 5.0f

void debugTheBuffer(GLParams p);

float map(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void debugPushRandomSpheresOnOneSideOfScreen(std::vector<raymarch::Renderable>& scene, std::uniform_real_distribution<float> d, std::mt19937 g, int count) {
  for (int i = 0; i < count; i++) {
    
    raymarch::Renderable sphere;
    sphere.type = raymarch::PrimitiveType::SPHERE;
    sphere.sphere.origin = glm::vec3(d(g), d(g), d(g));
    sphere.sphere.radius = 0.25f;
    sphere.color.r = map(d(g), RANDOM_MIN_RANGE, RANDOM_MAX_RANGE, 0, 255);
    sphere.color.g = map(d(g), RANDOM_MIN_RANGE, RANDOM_MAX_RANGE, 0, 255);
    sphere.color.b = map(d(g), RANDOM_MIN_RANGE, RANDOM_MAX_RANGE, 0, 255);
    sphere.color.a = 255;
    scene.push_back(sphere);
  }
}

__constant__ Camera camera;

int main()
{

  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_real_distribution<float> d(RANDOM_MIN_RANGE, RANDOM_MAX_RANGE);

  GLParams params;
  params.width = SCENE_WIDTH;
  params.height = SCENE_HEIGHT;
  cudaError_t error;
  Camera host_camera;
  Camera* device_camera;

  host_camera.position = glm::vec3(0, -10.0f, 0);
  host_camera.up = glm::vec3(0, 0, 1);
  host_camera.lookAt = glm::vec3(0, 0, 0);
  host_camera.fov = 60.0f;
  host_camera.aspect = (float)SCENE_WIDTH / (float)SCENE_HEIGHT;
  host_camera.near = 0.1f;
  host_camera.far = 100.0f;
  host_camera.aperture = APERTURE;
  host_camera.focalLength = FOCAL_DISTANCE;

  size_t numberOfRenderables = 8;
  std::vector<raymarch::Renderable> host_scene(numberOfRenderables);
 
  // Sphere 0 (Red Sphere - Large)
  host_scene[0].type = raymarch::PrimitiveType::SPHERE;
  host_scene[0].sphere.origin = glm::vec3(0.0f, 0.0f, -1.0f); // Centered at Z = -1
  host_scene[0].sphere.radius = 1.0f; // Large radius
  host_scene[0].color.r = 255;
  host_scene[0].color.g = 0;
  host_scene[0].color.b = 0;
  host_scene[0].color.a = 255;

  // Sphere 1 (Green Sphere - Medium)
  host_scene[1].type = raymarch::PrimitiveType::SPHERE;
  host_scene[1].sphere.origin = glm::vec3(-3.0f, 3.0f, -1.25f); // Positioned for shadow casting
  host_scene[1].sphere.radius = 0.75f; // Medium radius
  host_scene[1].color.r = 0;
  host_scene[1].color.g = 255;
  host_scene[1].color.b = 0;
  host_scene[1].color.a = 255;

  // Sphere 2 (Blue Sphere - Small)
  host_scene[2].type = raymarch::PrimitiveType::SPHERE;
  host_scene[2].sphere.origin = glm::vec3(3.5f, -2.0f, -1.5f); // Positioned separately
  host_scene[2].sphere.radius = 0.5f; // Small radius
  host_scene[2].color.r = 0;
  host_scene[2].color.g = 0;
  host_scene[2].color.b = 255;
  host_scene[2].color.a = 255;




  // First Triangle, scaled by 5 and moved down to Z = -2
  host_scene[4].type = raymarch::PrimitiveType::TRIANGLE;
  host_scene[4].triangle.v0 = glm::vec3(5, -5,  -2); // Bottom Right, scaled and moved
  host_scene[4].triangle.v1 = glm::vec3(-5, 5,  -2); // Top Left, scaled and moved
  host_scene[4].triangle.v2 = glm::vec3(-5, -5, -2); // Bottom Left, scaled and moved
  host_scene[4].color.r = 127;
  host_scene[4].color.g = 127;
  host_scene[4].color.b = 127;
  host_scene[4].color.a = 255;

  // Second Triangle, scaled by 5 and moved down to Z = -2
  host_scene[5].type = raymarch::PrimitiveType::TRIANGLE;
  host_scene[5].triangle.v0 = glm::vec3(5, -5, -2); // Bottom Right, scaled and moved
  host_scene[5].triangle.v1 = glm::vec3(5, 5,  -2); // Top Right, scaled and moved
  host_scene[5].triangle.v2 = glm::vec3(-5, 5, -2); // Top Left, scaled and moved
  host_scene[5].color.r = 127;
  host_scene[5].color.g = 127;
  host_scene[5].color.b = 127;
  host_scene[5].color.a = 255;

  host_scene[6].type = raymarch::PrimitiveType::ROUNDBOX;
  host_scene[6].roundBox.origin = glm::vec3(2, -1, -1.50);
  host_scene[6].roundBox.halfExtents = glm::vec3(.25, .25, .25);
  host_scene[6].roundBox.edgeWidth = 0.2f;
  host_scene[6].color.r = 255;
  host_scene[6].color.g = 0;
  host_scene[6].color.b = 255;
  host_scene[6].color.a = 255;


  host_scene[7].type = raymarch::PrimitiveType::BOXFRAME;
  host_scene[7].boxFrame.origin = glm::vec3(-3.3f, -2.5f, -1.0f);
  host_scene[7].boxFrame.halfExtents = glm::vec3(0.75f, 1, 1);
  host_scene[7].boxFrame.edgeWidth = 0.1f;
  host_scene[7].color.r = 0;
  host_scene[7].color.g = 255;
  host_scene[7].color.b = 255;
  host_scene[7].color.a = 255;



  if (ADD_RANDOM_SPHERES) {
    debugPushRandomSpheresOnOneSideOfScreen(host_scene, d, g, 100);
    numberOfRenderables += 100;
  }


  std::vector<raymarch::Light> host_lights;
  raymarch::Light* device_lights;

  raymarch::Light light;
  light.type = raymarch::LightType::DIRECTIONAL;
  light.color.r = 0.8f;
  light.color.g = 0.8f;
  light.color.b = 0.8f;
  light.directionalLight.direction = glm::normalize(glm::vec3(1.0f , -1.0f, 0.5f));


  //light.type = raymarch::LightType::AREA_PLANE;
  //light.color.r = 255;
  //light.color.g = 255;
  //light.color.b = 255;
  //light.areaLightPlane.origin = glm::vec3(0, 0, -2.0);
  //light.areaLightPlane.normal = glm::normalize(glm::vec3(0, 0, -1));
  //light.areaLightPlane.u = glm::vec3(2, 0, 0);
  //light.areaLightPlane.v = glm::vec3(0, 2, 0);



  host_lights.push_back(light);

  



  Octree* octree = nullptr;

  try {
    octree = new Octree(host_scene);
  }
  catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    delete octree;
    std::cout << "Failed to create octree" << std::endl;
    return -1;
  }


  if (DEBUG_VIZUALIZE_OCTREE) {

    raymarch::Color octreeColor;
    octreeColor.r = 255;
    octreeColor.g = 255;
    octreeColor.b = 255;
    octreeColor.a = 255;

    for (const OctreeNode& node : octree->tree) {
      raymarch::Renderable octreeNode;
      octreeNode.type = raymarch::PrimitiveType::BOXFRAME;
      octreeNode.boxFrame.halfExtents = node.bounds.halfExtents;
      octreeNode.boxFrame.origin = node.bounds.origin;
      octreeNode.boxFrame.edgeWidth = 0.0075f;
      octreeNode.color = octreeColor;
      host_scene.push_back(octreeNode);
      numberOfRenderables++;
    };

  }

  raymarch::Renderable* device_scene;


  if (!initGL(params)) {
    return -1;
  }

  OctreeNode* device_octree;
  raymarch::Ray* device_rays;
  raymarch::hitInfo* device_hits;


  std::cout << "allocating GPU memory..." << std::endl;

  CUDA_CHECK_ERROR(cudaMalloc(&device_camera, sizeof(Camera)));
  CUDA_CHECK_ERROR(cudaMemcpy(device_camera, &host_camera, sizeof(Camera), cudaMemcpyHostToDevice));

  CUDA_CHECK_ERROR(cudaMalloc(&device_scene, sizeof(raymarch::Renderable) * numberOfRenderables));
  CUDA_CHECK_ERROR(cudaMemcpy(device_scene, host_scene.data(), sizeof(raymarch::Renderable) * numberOfRenderables, cudaMemcpyHostToDevice));

  CUDA_CHECK_ERROR(cudaMalloc(&device_lights, sizeof(raymarch::Light) * host_lights.size()));
  CUDA_CHECK_ERROR(cudaMemcpy(device_lights, host_lights.data(), sizeof(raymarch::Renderable) * host_lights.size(), cudaMemcpyHostToDevice));


  CUDA_CHECK_ERROR(cudaMalloc(&device_octree, sizeof(OctreeNode) * octree->size()));
  CUDA_CHECK_ERROR(cudaMemcpy(device_octree, octree->data(), sizeof(OctreeNode) * octree->size(), cudaMemcpyHostToDevice));




  cudaGraphicsResource* cudaPixelBufferResource;
  CUDA_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cudaPixelBufferResource, params.pixelBuffer, cudaGraphicsMapFlagsWriteDiscard));



  unsigned char* devPtr;
  size_t size;

  dim3 blockSize(16, 16); // A common choice, but adjust based on your needs and GPU architecture
  dim3 gridSize((SCENE_WIDTH + blockSize.x - 1) / blockSize.x, (SCENE_HEIGHT + blockSize.y - 1) / blockSize.y);

  size_t num_threads = blockSize.x * blockSize.y * blockSize.z * gridSize.x * gridSize.y * gridSize.z;
  size_t seed = 512;
  curandState* devStates;
  CUDA_CHECK_ERROR(cudaMalloc(&devStates, num_threads * sizeof(curandState)));

 
  
  CUDA_TIME_IT(raymarch::initRandState, gridSize, blockSize, "Random state init time", devStates, seed);




  CUDA_CHECK_ERROR(cudaDeviceSynchronize()); // Optional, for synchronization/debugging

  if (ENABLE_ANTI_ALIASING) {
    CUDA_CHECK_ERROR(cudaMalloc(&device_rays, sizeof(raymarch::Ray) * num_threads * SQRT_SAMPLE_PER_PIXEL * SQRT_SAMPLE_PER_PIXEL));
    CUDA_CHECK_ERROR(cudaMalloc(&device_hits, sizeof(raymarch::hitInfo) * num_threads * SQRT_SAMPLE_PER_PIXEL * SQRT_SAMPLE_PER_PIXEL));
  }
  else {
    CUDA_CHECK_ERROR(cudaMalloc(&device_rays, sizeof(raymarch::Ray) * num_threads));
    CUDA_CHECK_ERROR(cudaMalloc(&device_hits, sizeof(raymarch::hitInfo) * num_threads));
  }


  raymarch::MarchingOrders marchingOrders;
  marchingOrders.camera = device_camera;
  marchingOrders.rays = device_rays;
  marchingOrders.scene = device_scene;
  marchingOrders.lights = device_lights;
  marchingOrders.lightSize = host_lights.size();
  marchingOrders.octree = device_octree;
  marchingOrders.octreeSize = octree->size();
  marchingOrders.sceneSize = numberOfRenderables;
  marchingOrders.randState = devStates;
  marchingOrders.withAntiAliasing = ENABLE_ANTI_ALIASING;
  marchingOrders.renderNormals = RENDER_NORMALS;
  marchingOrders.sqrtSamplesPerPixel = SQRT_SAMPLE_PER_PIXEL;
  marchingOrders.height = SCENE_HEIGHT;
  marchingOrders.width = SCENE_WIDTH;
  marchingOrders.hitBuffer = device_hits;


  cudaEvent_t start, stop;
  cudaEvent_t pStart, pStop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&pStart);
  float milliseconds = 0;
  cudaEventCreate(&pStop);

  cudaEventRecord(start);


  CUDA_TIME_IT(raymarch::initRayKernel,gridSize, blockSize ,"Ray init time", marchingOrders );




  CUDA_TIME_IT(raymarch::updateHitBufferKernel, gridSize, blockSize, "HitBuffer init time", marchingOrders );


  // Render here

  CUDA_CHECK_ERROR(cudaGraphicsMapResources(1, &cudaPixelBufferResource, 0));

  CUDA_CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPixelBufferResource));

  CUDA_TIME_IT(raymarch::computeRayMarchedColorsKernel, gridSize, blockSize, "Color computation time", devPtr, marchingOrders)

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "With Anti-Aliasing : " << (ENABLE_ANTI_ALIASING ? "yes": "no");
  if(ENABLE_ANTI_ALIASING) std::cout << "; SPP: " << (SQRT_SAMPLE_PER_PIXEL * SQRT_SAMPLE_PER_PIXEL);
  std::cout << "; O-DEPTH: " << MAX_OCTREE_DEPTH;  
  std::cout << "; Number of renderables: " << numberOfRenderables;  
  std::cout << "; Total execution time: " << milliseconds << " milliseconds\n";


  CUDA_CHECK_ERROR(cudaGraphicsUnmapResources(1, &cudaPixelBufferResource, 0));

  // Loop until the user closes the window
  while (!glfwWindowShouldClose(params.window))
  {
    //debugTheBuffer(params);

    updateTexture(params);

    render(params);


  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  CUDA_CHECK_ERROR(cudaFree(device_camera));
  CUDA_CHECK_ERROR(cudaFree(device_scene));
  CUDA_CHECK_ERROR(cudaFree(devStates));
  CUDA_CHECK_ERROR(cudaFree(device_octree));
  CUDA_CHECK_ERROR(cudaFree(device_rays));
  CUDA_CHECK_ERROR(cudaFree(device_hits));
  CUDA_CHECK_ERROR(cudaFree(device_lights));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(pStart);
  cudaEventDestroy(pStop);


  glfwDestroyWindow(params.window);
  glfwTerminate(); // Cleanup and close the window once the loop is exited
  delete octree;

  return 0;
}

void debugTheBuffer(GLParams p)
{

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, p.pixelBuffer);
  unsigned char* ptr = (unsigned char*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_READ_ONLY);

  if (ptr) {
    // Just as an example, let's check the first pixel
    std::cout << "First pixel RGBA values: ";
    std::cout << (unsigned int)ptr[0] << ", " // Red
      << (unsigned int)ptr[1] << ", " // Green
      << (unsigned int)ptr[2] << ", " // Blue
      << (unsigned int)ptr[3] << std::endl; // Alpha
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
  }
  else {
    std::cout << "Failed to map the buffer" << std::endl;
  }

}


