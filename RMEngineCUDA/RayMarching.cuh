#ifndef RAY_MARCHING_CUH
#define RAY_MARCHING_CUH



#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <glm/glm.hpp>
#include "Camera.cuh"
#include "Octree.h"
#include "Renderable.h"

#define MAX_FLOAT 3.402823466e+38F

namespace raymarch {

  struct hitInfo {
    bool hit;
    glm::vec3 hitPoint;
    glm::vec3 normal;
    Renderable object;
    Ray incidentRay;
  };

  struct lightHitInfo {
    bool hit;
    glm::vec3 hitPoint;
    Light light;
  };


  struct MarchingOrders {
    Camera* camera;
    Ray* rays;
    hitInfo* hitBuffer;
    Renderable* scene;
    Light* lights;
    OctreeNode* octree;
    size_t octreeSize;
    size_t sceneSize;
    size_t lightSize;
    curandState* randState;
    int width;
    int height;
    int sqrtSamplesPerPixel;
    bool withAntiAliasing;
    bool renderNormals;
    bool useDof;
    glm::vec3 bgColor;
  };


#pragma region RAYMARCH DEVICE FUNCTIONS
  __device__ void calculateRay(Ray& ray, float x, float y, Camera* camera, int width, int height);
  __device__ void calculateDOFRay(Ray& ray, float u, float v, curandState* state, const MarchingOrders& orders);
  __device__ glm::vec3 computeColorFromRay(Ray& ray, const MarchingOrders& marchingOrders);

  __device__ glm::vec3 computeColorFromHitInfo(int hitIndex, const MarchingOrders& marchingOrders);

  __device__ hitInfo computeHitInfo(const Ray& ray, const MarchingOrders& marchingOrders, bool debug);

  __device__ glm::vec3 accumulateLightBounce(hitInfo hit, const MarchingOrders& marchingOrders, int depth);
  __device__ lightHitInfo computeLightHitInfo(const Ray& ray, const Light& light);

#pragma endregion

#pragma region UTILITY FUNCTIONS
  __device__ void clamp(glm::vec3& vec, float min, float max);
  __device__ glm::vec3 colorToColorVec(const Color& color);
  __device__ Color colorVecToColor(glm::vec3& colorVec);
  __device__ float map(float x, float in_min, float in_max, float out_min, float out_max);
  __device__ void correctGamma(glm::vec3& color);
  __device__ float clamp(float x, float min, float max);
  __device__ float dot2(const glm::vec3& v);
  __device__ glm::vec3 getLightDirection(const Light& light, const glm::vec3& hitPoint);
  __device__ void perturbRayDirection(Ray& ray, curandState* randState, float maxAngle);
  __device__ glm::vec2 randomInUnitDisk(curandState* state);

#pragma endregion

#pragma region SHADING FUNCTIONS
  __device__ glm::vec3 calculatePhongShading(const hitInfo& hit, const Light& light);
  __device__ glm::vec3 calculateNormalShading(glm::vec3& color, const glm::vec3& normal);
#pragma endregion

#pragma region SDF FUNCTIONS

  __device__ float sdf(glm::vec3 checkPoint, Renderable* scene, size_t sceneSize, Renderable& hitObject);
  __device__ float sphereSDF(glm::vec3 p, const Sphere& s);
  __device__ float boxSDF(glm::vec3 checkPoint, glm::vec3 halfExtents, glm::vec3 origin);
  __device__ float boxFrameSDF(glm::vec3 checkPoint, glm::vec3 halfExtents, glm::vec3 origin, float edgeWidth);
  __device__ float roundBoxSDF(glm::vec3 checkPoint, glm::vec3 halfExtents, glm::vec3 origin, float edgeWidth);
  __device__ float triangleSDF(glm::vec3 checkPoint, const Triangle& t);
  __device__ float discSDF(glm::vec3 checkPoint, const Disc& d);

  __device__ float lightSDF(glm::vec3 checkPoint, const Light* light);
  __device__ float pointLightSDF(glm::vec3 checkPoint, const PointLight& light);
  __device__ float directionalLightSDF(glm::vec3 checkPoint, const DirectionalLight& light);
  //todo __device__ float spotLightSDF(glm::vec3 checkPoint, const SpotLight& light);
  //todo__device__ float sphereLightSDF(glm::vec3 checkPoint, const SphereLight& light);
  __device__ float areaLightSDF(glm::vec3 checkPoint, const AreaLightPlane& light);
  __device__ float discLightSDF(glm::vec3 checkPoint, const AreaLightDisc& light);




#pragma endregion

#pragma region NORMALS FUNCTIONS
  __device__ glm::vec3 getNormal(const Renderable& hitObject, glm::vec3 p);
  __device__ glm::vec3 boxNormal(glm::vec3 p, glm::vec3 halfExtents, glm::vec3 origin);
  __device__ glm::vec3 triangleNormal(glm::vec3 p, const Triangle& t);
  __device__ glm::vec3 roundBoxNormal(glm::vec3 p, glm::vec3 halfExtents, glm::vec3 origin, float edgeWidth);
  __device__ glm::vec3 boxFrameNormal(glm::vec3 p, glm::vec3 halfExtents, glm::vec3 origin, float edgeWidth);
  __device__ glm::vec3 estimateNormal(glm::vec3 p, const Renderable& hitObject);
  __device__ glm::vec3 discNormal(glm::vec3 p, const Disc& d);


#pragma endregion

#pragma region GLOBAL KERNELS

  __global__ void computeRayMarchedColorsKernel(unsigned char* pixelBuffer, MarchingOrders marchingOrders);

  __global__ void initRandState(curandState* randState, size_t seed);

  __global__ void debugTheRandom(curandState* randState);

  __global__ void initRayKernel(MarchingOrders marchingOrders);

  __global__ void updateHitBufferKernel(const MarchingOrders marchingOrders);

#pragma endregion
  
#pragma region OCTREE FUNCTIONS
  __device__ bool isRayInOctreeBounds(const Ray& ray, const OctreeNode* tree, size_t treeSz, bool debug);
  __device__ bool doesRayHitNode(const Ray& ray, const OctreeNode& node, const OctreeNode* tree, float tmax, bool debug);
  __device__ bool isNodeLeaf(const OctreeNode& node, bool debug);
#pragma endregion


}  // namespace raymarch

#endif // !RAY_MARCHING_CUH