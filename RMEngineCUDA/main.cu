

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
#include "BlueprintDecoder.h"



#define SCENE_WIDTH 1280
#define SCENE_HEIGHT 720

//#define SCENE_WIDTH 800
//#define SCENE_HEIGHT 600

#define ENABLE_ANTI_ALIASING true
#define RENDER_NORMALS false
#define DEBUG_VIZUALIZE_OCTREE false
#define RENDER_DEPTH false

#define WITH_DOF false
#define APERTURE 0.3f
#define FOCAL_DISTANCE 10.05f

#define ADD_RANDOM_SPHERES false
#define SQRT_SAMPLE_PER_PIXEL 8

#define RANDOM_MIN_RANGE 0.0f
#define RANDOM_MAX_RANGE 5.0f

#if RENDER_DEPTH == true

    #undef ENABLE_ANTI_ALIASING
    #define ENABLE_ANTI_ALIASING false
    #undef RENDER_NORMALS
    #define RENDER_NORMALS false
    #undef WITH_DOF
    #define WITH_DOF false
    #undef ADD_RANDOM_SPHERES
    #define ADD_RANDOM_SPHERES false
    #undef SQRT_SAMPLE_PER_PIXEL
    #define SQRT_SAMPLE_PER_PIXEL 1
  
#endif


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
  
  BlueprintDecoder *pdecoder;
  try {
    pdecoder = new BlueprintDecoder("Demo.scene", false);
  }
  catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }
  
  BlueprintDecoder& decoder = *pdecoder;


  std::vector<raymarch::Renderable> host_scene = decoder.getRenderables();
  std::vector<raymarch::Light> host_lights = decoder.getLights();
  SceneSettings settings = decoder.getSettings();

  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_real_distribution<float> d(RANDOM_MIN_RANGE, RANDOM_MAX_RANGE);

  GLParams params;
  params.width = settings.width;
  params.height = settings.height;
  cudaError_t error;
  Camera host_camera = decoder.getCamera();
  Camera* device_camera;


  if (ADD_RANDOM_SPHERES) {
    debugPushRandomSpheresOnOneSideOfScreen(host_scene, d, g, 100);
    //numberOfRenderables += 100;
  }

  raymarch::Light* device_lights;
  
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
      //numberOfRenderables++;
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

  CUDA_CHECK_ERROR(cudaMalloc(&device_scene, sizeof(raymarch::Renderable) * host_scene.size()));
  CUDA_CHECK_ERROR(cudaMemcpy(device_scene, host_scene.data(), sizeof(raymarch::Renderable) * host_scene.size(), cudaMemcpyHostToDevice));

  CUDA_CHECK_ERROR(cudaMalloc(&device_lights, sizeof(raymarch::Light) * host_lights.size()));
  CUDA_CHECK_ERROR(cudaMemcpy(device_lights, host_lights.data(), sizeof(raymarch::Renderable) * host_lights.size(), cudaMemcpyHostToDevice));


  CUDA_CHECK_ERROR(cudaMalloc(&device_octree, sizeof(OctreeNode) * octree->size()));
  CUDA_CHECK_ERROR(cudaMemcpy(device_octree, octree->data(), sizeof(OctreeNode) * octree->size(), cudaMemcpyHostToDevice));




  cudaGraphicsResource* cudaPixelBufferResource;
  CUDA_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cudaPixelBufferResource, params.pixelBuffer, cudaGraphicsMapFlagsWriteDiscard));



  unsigned char* devPtr;
  size_t size;

  dim3 blockSize(16, 16); // A common choice, but adjust based on your needs and GPU architecture
  dim3 gridSize((settings.width + blockSize.x - 1) / blockSize.x, (settings.height + blockSize.y - 1) / blockSize.y);

  size_t num_threads = blockSize.x * blockSize.y * blockSize.z * gridSize.x * gridSize.y * gridSize.z;
  size_t seed = 512;
  curandState* devStates;
  CUDA_CHECK_ERROR(cudaMalloc(&devStates, num_threads * sizeof(curandState)));

 
  
  CUDA_TIME_IT(raymarch::initRandState, gridSize, blockSize, "Random state init time", devStates, seed);




  CUDA_CHECK_ERROR(cudaDeviceSynchronize()); // Optional, for synchronization/debugging

  if (ENABLE_ANTI_ALIASING) {
    CUDA_CHECK_ERROR(cudaMalloc(&device_rays, sizeof(raymarch::Ray) * num_threads * settings.antiAliasingQuality * settings.antiAliasingQuality));
    CUDA_CHECK_ERROR(cudaMalloc(&device_hits, sizeof(raymarch::hitInfo) * num_threads * settings.antiAliasingQuality * settings.antiAliasingQuality));
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
  marchingOrders.sceneSize = host_scene.size();
  marchingOrders.randState = devStates;
  marchingOrders.withAntiAliasing = settings.enableAntiAliasing;
  marchingOrders.renderNormals = RENDER_NORMALS;
  marchingOrders.sqrtSamplesPerPixel = settings.antiAliasingQuality;
  marchingOrders.height = settings.height;
  marchingOrders.width = settings.width;
  marchingOrders.hitBuffer = device_hits;
  marchingOrders.useDof = settings.enableDoF;


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

  if (RENDER_DEPTH) {
    CUDA_TIME_IT(raymarch::createDepthBufferKernel, gridSize, blockSize, "Color computation time", devPtr, marchingOrders);
  }
  else {
    CUDA_TIME_IT(raymarch::computeRayMarchedColorsKernel, gridSize, blockSize, "Color computation time", devPtr, marchingOrders);

  }


  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "With Anti-Aliasing : " << (settings.enableAntiAliasing ? "yes": "no");
  if(ENABLE_ANTI_ALIASING) std::cout << "; SPP: " << (settings.antiAliasingQuality * settings.antiAliasingQuality);
  std::cout << "; O-DEPTH: " << MAX_OCTREE_DEPTH;  
  std::cout << "; Number of renderables: " << host_scene.size();  
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
  delete pdecoder;

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


