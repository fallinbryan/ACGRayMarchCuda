#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "RayMarching.cuh"
#include "Octree.h"

#pragma region GLOBAL KERNELS


__global__ void raymarch::computeRayMarchedColorsKernel(unsigned char* pixelBuffer, raymarch::MarchingOrders marchingOrders) {


  bool debug = false;
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  

  if (x >= marchingOrders.width || y >= marchingOrders.height) return;

  //debug = (x == 676 && y == 353);

  int curandId = threadIdx.x + blockIdx.x * blockDim.x;
  int sqrtSamplesPerPixel = marchingOrders.sqrtSamplesPerPixel;

  glm::vec3 lightDirection = glm::normalize(glm::vec3(1, -1, -1));
  
  raymarch::Color lightColor;
  lightColor.r = 200;
  lightColor.g = 200;
  lightColor.b = 200;
  lightColor.a = 255;


  glm::vec3 vRenderColor = glm::vec3(0);
  raymarch::Color renderColor;
  glm::vec3 bgColor;

  glm::vec3 sumColor = glm::vec3(0, 0, 0);

  if (marchingOrders.withAntiAliasing) {

    int testRayId = (y * marchingOrders.width + x) * sqrtSamplesPerPixel * sqrtSamplesPerPixel;
    raymarch:Ray testRay = marchingOrders.rays[testRayId];
    bgColor.r = raymarch::map(testRay.direction.x, -1.0f, 1.0f, 0.0f, 1.0f);
    bgColor.g = raymarch::map(testRay.direction.y, -1.0f, 1.0f, 0.0f, 1.0f);
    bgColor.b = raymarch::map(testRay.direction.z, -1.0f, 1.0f, 0.0f, 1.0f);
    marchingOrders.bgColor = bgColor;
    if (raymarch::isRayInOctreeBounds(testRay, marchingOrders.octree, marchingOrders.octreeSize, debug))
    {

      for (int i = 0; i < sqrtSamplesPerPixel; i++)
      {
        for (int j = 0; j < sqrtSamplesPerPixel; j++)
        {

          int rayId = (y * marchingOrders.width + x) * sqrtSamplesPerPixel * sqrtSamplesPerPixel + i * sqrtSamplesPerPixel + j;
          raymarch::Ray ray = marchingOrders.rays[rayId];


          glm::vec3 colorvec;

          bgColor.r = raymarch::map(ray.direction.x, -1.0f, 1.0f, 0.0f, 1.0f);
          bgColor.g = raymarch::map(ray.direction.y, -1.0f, 1.0f, 0.0f, 1.0f);
          bgColor.b = raymarch::map(ray.direction.z, -1.0f, 1.0f, 0.0f, 1.0f);
          //bgColor.a = 255;
            marchingOrders.bgColor = bgColor;
            colorvec = raymarch::computeColorFromRay(ray, marchingOrders);
            sumColor += colorvec;
            

          }
        }
      vRenderColor = sumColor / (float)(sqrtSamplesPerPixel * sqrtSamplesPerPixel);
    }
    else
    {
      vRenderColor = bgColor;
    }


  }
  else
  {
    int rayId = x + y * marchingOrders.width;
    raymarch::Ray ray = marchingOrders.rays[rayId];

    bgColor.r = raymarch::map(ray.direction.x, -1.0f, 1.0f, 0.0f, 1.0f);
    bgColor.g = raymarch::map(ray.direction.y, -1.0f, 1.0f, 0.0f, 1.0f);
    bgColor.b = raymarch::map(ray.direction.z, -1.0f, 1.0f, 0.0f, 1.0f);
    //bgColor.a = 255;
    marchingOrders.bgColor = bgColor;
    if (raymarch::isRayInOctreeBounds(ray, marchingOrders.octree, marchingOrders.octreeSize, debug))
    {
      vRenderColor = raymarch::computeColorFromRay(ray, marchingOrders);

      //if(vRenderColor != glm::vec3(0) && vRenderColor != bgColor)
      //{
      // //printf("render color: %f, %f, %f\n", vRenderColor.x, vRenderColor.y, vRenderColor.z);
      //  glm::vec3 temp = vRenderColor;
      //  raymarch::correctGamma(temp);
      // //printf("corrected color: %f, %f, %f\n", temp.x, temp.y, temp.z);
      //}
    }
    else
    {
      vRenderColor = bgColor;
    }
  }


    raymarch::correctGamma(vRenderColor);

    renderColor = raymarch::colorVecToColor(vRenderColor);

    int index = (x + y * marchingOrders.width) * 4; // Assuming 4 bytes per pixel (RGBA)
    pixelBuffer[index] = renderColor.r /* red value */;
    pixelBuffer[index + 1] = renderColor.g /* green value */;
    pixelBuffer[index + 2] = renderColor.b /* blue value */;
    pixelBuffer[index + 3] = renderColor.a; // Alpha value


}

__global__ void raymarch::initRayKernel(MarchingOrders marchingOrders) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= marchingOrders.width || y >= marchingOrders.height) return;
  
  if (!marchingOrders.withAntiAliasing) {

    raymarch::Ray ray;
    int rayId = x + y * marchingOrders.width;
    raymarch::calculateRay(ray, x, y, marchingOrders.camera, marchingOrders.width, marchingOrders.height);
    ray.x = x;
    ray.y = y;
    marchingOrders.rays[rayId] = ray;
  } //else we will jitter the rays for anit-aliasing
  else
  {
    int sqrtSamplesPerPixel = marchingOrders.sqrtSamplesPerPixel;
    int curandId = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = marchingOrders.randState[curandId];

    for (int i = 0; i < sqrtSamplesPerPixel; i++) {
      for (int j = 0; j < sqrtSamplesPerPixel; j++) {

        int rayId = (y * marchingOrders.width + x) * sqrtSamplesPerPixel * sqrtSamplesPerPixel + i * sqrtSamplesPerPixel + j;

        float jitterX = curand_uniform(&localState) * 10;
        float jitterY = curand_uniform(&localState) * 10;

        float u = x + (jitterX / sqrtSamplesPerPixel);
        float v = y + (jitterY / sqrtSamplesPerPixel);

        raymarch::Ray ray;
        ray.x = u;
        ray.y = v;
        if (marchingOrders.useDof) {
          marchingOrders.randState[curandId] = localState;
          raymarch::calculateDOFRay(ray, u, v, &localState, marchingOrders);
        }
        else {
          raymarch::calculateRay(ray, u, v, marchingOrders.camera, marchingOrders.width, marchingOrders.height);

        }
        marchingOrders.rays[rayId] = ray;

      }
    }
    marchingOrders.randState[curandId] = localState;
  }
}

__global__ void raymarch::debugTheRandom(curandState* randState) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (x >= 10 || y >= 10) return;

  curandState localState = randState[id];
 //printf("x: %d, y: %d, random: %f\n", x, y, curand_uniform(&localState));
  randState[id] = localState;
}

__global__ void raymarch::initRandState(curandState* randState, size_t seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &randState[id]);
}
#pragma endregion

#pragma region RAYMARCH DEVICE FUNCTIONS

__device__ void raymarch::calculateRay(Ray& ray, float x, float y, Camera* camera, int width, int height) {
  glm::vec2 pixel = glm::vec2(x, y);

  camera->normalizeCoordinates(pixel, width, height);
  camera->scaleCoordinates(pixel);

  glm::vec3 rayDirection = glm::normalize(glm::vec3(pixel.x, pixel.y, -1.0f));

  camera->toWorldSpace(rayDirection);
  ray.origin = camera->position;
  ray.direction = rayDirection;

}

__device__ void raymarch::calculateDOFRay(Ray& ray, float pixelX, float pixelY, curandState* state, const MarchingOrders& orders) {

  // Calculate camera basis vectors
  glm::vec3 forward = glm::normalize(orders.camera->lookAt - orders.camera->position);
  glm::vec3 right = glm::normalize(glm::cross(forward, orders.camera->up));
  glm::vec3 up = glm::cross(right, forward);

  // Lens radius for DoF
  float lensRadius = orders.camera->aperture / 2.0f;

  // Sample point on lens
  glm::vec2 diskSample = randomInUnitDisk(state) * lensRadius;
  glm::vec3 lensOffset = right * diskSample.x + up * diskSample.y;

  // Adjust pixel coordinates for aspect ratio, FOV, and to NDC
  glm::vec2 pixel(pixelX, pixelY);
  orders.camera->normalizeCoordinates(pixel, orders.width, orders.height);
  orders.camera->scaleCoordinates(pixel);

  // Calculate ray direction in camera space
  glm::vec3 rayDirection(pixel.x, pixel.y, -1.0f); // Assume camera looks towards negative Z in camera space
  // Convert ray direction to world space
  orders.camera->toWorldSpace(rayDirection);

  // Calculate the point where the ray intersects the focal plane
  float focalDistance = orders.camera->focalLength; // Use focalLength as focal distance
  glm::vec3 focalPoint = orders.camera->position + rayDirection * focalDistance;

  // Set up the ray
  ray.origin = orders.camera->position + lensOffset; // Apply lens offset for DoF
  ray.direction = glm::normalize(focalPoint - ray.origin);
}


__device__ raymarch::hitInfo raymarch::computeHitInfo(const Ray& ray, const MarchingOrders& marchingOrders) {

 ////printf("computing hit info for ray: %f, %f\n", ray.x, ray.y);

  raymarch::Renderable hitObject;
  raymarch::hitInfo hi;
  hi.hit = false;
  glm::vec3 checkPoint = ray.origin;

  float maxDistance = 100.0f;
  float totalDistance = 0.0f;
  float maxSteps = 1000.0f;
  float epsilon = 0.00001f;
  float distance = 0.0f;

  for (int i = 0; i < maxSteps; i++) {
    if (totalDistance > maxDistance)
    {
      //printf("total distance %f exceeds hit max distance %f at step %d\n", totalDistance, maxDistance, i);
      break;
    }
    distance = raymarch::sdf(checkPoint, marchingOrders.scene, marchingOrders.sceneSize, hitObject);
    checkPoint += ray.direction * distance;
    totalDistance += distance;
    if (distance < epsilon) {
      hi.hit = true;
      hi.hitPoint = checkPoint;
      hi.object = hitObject;
      hi.incidentRay.direction = ray.direction;
      hi.incidentRay.origin = ray.origin;
      hi.normal = raymarch::getNormal(hitObject, checkPoint);
      break;
    }
  }
  //if(hi.hit)
  // //printf("hit: %d\n", hi.hit ? 1 : 0);
  return hi;
}

__device__ glm::vec3 raymarch::computeColorFromRay(Ray& ray, const MarchingOrders& marchingOrders) {
  

  glm::vec3 color;
  raymarch::Ray bouncRay;
  float epsilon = 0.0001f;
  color.r = marchingOrders.bgColor.r;
  color.g = marchingOrders.bgColor.g;
  color.b = marchingOrders.bgColor.b;

  hitInfo hit = raymarch::computeHitInfo(ray, marchingOrders);
  if(!hit.hit) return color;
  //color = raymarch::accumulateLightBounce(hit, marchingOrders, 0);
  float shadowFactor = 1.0f;
  bouncRay.direction = raymarch::getLightDirection(marchingOrders.lights[0], hit.hitPoint);
  bouncRay.origin = hit.hitPoint + hit.normal * epsilon;
  perturbRayDirection(bouncRay, marchingOrders.randState, PI/2.0f);
  raymarch::hitInfo bouncHit = raymarch::computeHitInfo(bouncRay, marchingOrders);
  if (bouncHit.hit)
  {
    shadowFactor = 0.5f;
  };
  color = raymarch::calculatePhongShading(hit, marchingOrders.lights[0]);
  



  return color * shadowFactor;
}

__device__ glm::vec3 raymarch::accumulateLightBounce(hitInfo hit, const MarchingOrders& marchingOrders, int depth)
{
  return raymarch::calculatePhongShading(hit, marchingOrders.lights[0]); //color;
  glm::vec3 color = glm::vec3(0);
  float epsilon = 0.00001f;
  float shadowFactor = 1.0f;
  for (int i = 0; i < marchingOrders.lightSize; i++) {}
    raymarch::Ray bouncRay;
    //bouncRay.direction = raymarch::getLightDirection(marchingOrders.lights[0], hit.hitPoint);
    //bouncRay.origin = hit.hitPoint + bouncRay.direction * epsilon;
  //  raymarch::hitInfo bouncHit = raymarch::computeHitInfo(bouncRay, marchingOrders);
  //  raymarch::lightHitInfo lhi = raymarch::computeLightHitInfo(bouncRay, marchingOrders.lights[i]);
  //  if (true) {
  //
  //    color = raymarch::calculatePhongShading(hit, marchingOrders.lights[i]);

  //  }
  //}
}



__device__ raymarch::lightHitInfo raymarch::computeLightHitInfo(const raymarch::Ray& ray, const raymarch::Light& light) {
  raymarch::lightHitInfo lhi;
  lhi.hit = false;
  //printf("computing light hit info for light type: %d\n", light.type);


  if (light.type == raymarch::LightType::DIRECTIONAL) {
    lhi.hit = true;
    lhi.hitPoint = ray.origin + ray.direction * 1000.0f;
    lhi.light = light;
    return lhi;
  };

  glm::vec3 checkPoint = ray.origin;

  float maxDistance = 100.0f;
  float totalDistance = 0.0f;
  float maxSteps = 1000.0f;
  float epsilon = 0.00001f;

  for (int i = 0; i < maxSteps; i++) {
    if (totalDistance > maxDistance) break;
    float distance = raymarch::lightSDF(checkPoint, &light);
    checkPoint += ray.direction * distance;
    totalDistance += distance;
    if (distance < epsilon) {
      lhi.hit = true;
      lhi.hitPoint = checkPoint;
      lhi.light = light;
      break;
    }
  }
  return lhi;
}

#pragma endregion

#pragma region OCTREE DEVICE FUNCTIONS



__device__ bool raymarch::isRayInOctreeBounds(const Ray& ray, const OctreeNode* tree, size_t treeSz, bool debug) {
  return true;
  bool returnValue = false;
  
  OctreeNode node = tree[0];
  returnValue = raymarch::doesRayHitNode(ray, node, tree, 1000.0f, debug);
  if(!returnValue) return false;

  // at this point, we know the ray hits the root node, so we can start checking the children
  // since the tree is in a flat array, we can use the parent index to find the children
  // and then in while loop, we can keep checking the children until we find a leaf node
  // this will avoid recursion and allow us to use the GPU to traverse the tree
  while(!raymarch::isNodeLeaf(node, debug)) {
    bool hitChild = false;
    /// i have  set a debug pixel that I know should hit an object, but it is not hitting the object
    if (debug) {
     //printf("node bounds: %f, %f, %f\n", node.bounds.halfExtents.x, node.bounds.halfExtents.y, node.bounds.halfExtents.z);
     //printf("node position: %f, %f, %f\n", node.bounds.origin.x, node.bounds.origin.y, node.bounds.origin.z);
     //printf("node idx: %d\n", node.nodeIdx);
     //printf("node is leaf: %d\n", raymarch::isNodeLeaf(node, debug) ? 1 : 0);
     //printf("node has data: %d\n", node.containsData ? 1 : 0);
    }
    
    for (int i = 0; i < 8; i++) 
    {
      
      if (node.childIndices[i] != -1) 
      {
        if(debug) printf("child: %d\n", node.childIndices[i]);
        
        OctreeNode child = tree[node.childIndices[i]];
       

        if (debug) {
         //printf("ParentChildIdex: %d child %d bounds: %f, %f, %f\n", i, child.nodeIdx, child.bounds.halfExtents.x, child.bounds.halfExtents.y, child.bounds.halfExtents.z);
         //printf("child %d position: %f, %f, %f\n", child.nodeIdx, child.bounds.origin.x, child.bounds.origin.y, child.bounds.origin.z);
         //printf("Child Parent Idex: %d\n", child.parentIdx);
         //printf("child %d is node leaf: %d\n", child.nodeIdx, raymarch::isNodeLeaf(child, debug) ? 1 : 0);
         //printf("child %d has data: %d\n", child.nodeIdx, child.containsData ? 1 : 0);
        }

        if (raymarch::doesRayHitNode(ray, child, tree, 1000.0f, debug)) 
        {
          node = child;
          hitChild = true;
         
          if (debug) 
          {
           //printf("hit child\n");
           //printf("child is node leaf: %d\n", raymarch::isNodeLeaf(child, debug) ? 1 : 0);
          }
          
          break;
        }

        if(debug) printf("did not hit child %d\n", child.nodeIdx);
      }
      if(debug) printf("Parent.childIdx %d is -1, no child was added\n", i);
    }
    if (!hitChild) {
      if (debug)//printf("did not hit any child, leaving the loop as no hit to octree\n");
      return false;
    }
  }

  return returnValue;
}

__device__ bool raymarch::doesRayHitNode(const raymarch::Ray& ray, const OctreeNode& node, const OctreeNode* tree, float tmax, bool debug) {
  float epsilon = 0.00001f;
  float totalDistance = 0.0f;
  int maxSteps = 800;
  if(debug) printf("checking node %d\n", node.nodeIdx);
  for (int step = 0; step < maxSteps; step++) {

    if (totalDistance > tmax)
    {
      if(debug) printf("total distance %f exceeds hit max distance %f\n",totalDistance, tmax);
      return false;
    }

    float distance = raymarch::boxSDF(ray.origin + ray.direction * totalDistance, node.bounds.halfExtents, node.bounds.origin);
    if (debug)//printf("distance: %f at step: %d\n", distance, step);
    totalDistance += distance;
    
    if (distance < epsilon) {
      if(debug) printf("hit within max steps %d with epsilon %f, totalDistance %f\n", maxSteps, epsilon, totalDistance);
      return true;  // Hit the leaf node
    }
    
  }
  if(debug) printf("no hit within max steps %d with epsilon %f, totalDistance %f\n", maxSteps, epsilon, totalDistance);
  return false;  // No hit within max steps
}

__device__ bool raymarch::isNodeLeaf(const OctreeNode& node, bool debug) {
  // a node is a leaf if it contains data and all 8 of the children are -1
  bool result = true;
  for (int i = 0; i < 8; i++) {
    if (node.childIndices[i] != -1) {
      result = false;
      break;
    }
  }
  return result && node.containsData;
}

#pragma endregion

#pragma region UTILITY FUNCTIONS
__device__ void raymarch::clamp(glm::vec3& vec, float min, float max) {
  vec.x = fminf(fmaxf(vec.x, min), max);
  vec.y = fminf(fmaxf(vec.y, min), max);
  vec.z = fminf(fmaxf(vec.z, min), max);
}

__device__ glm::vec3 raymarch::colorToColorVec(const Color& color) {
  return glm::vec3(color.r, color.g, color.b) / 255.0f;
}

__device__ raymarch::Color raymarch::colorVecToColor(glm::vec3& colorVec) {
  raymarch::Color color;
  color.r = colorVec.x * 255;
  color.g = colorVec.y * 255;
  color.b = colorVec.z * 255;
  color.a = 255;
  return color;
}

__device__ float raymarch::map(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

__device__ void raymarch::correctGamma(glm::vec3& color) {
  float exp = 1.0f / 2.2f;
  if (color.r > 1.0f) color.r = 1.0f;
  if (color.g > 1.0f) color.g = 1.0f;
  if (color.b > 1.0f) color.b = 1.0f;

  if (color.r < 0.0031308f) {
    color.r = 12.92f * color.r;
  }
  else {
    color.r = 1.055f * pow(color.r, exp) - 0.055f;
  }
  if (color.g < 0.0031308f) {
    color.g = 12.92f * color.g;
  }
  else {
    color.g = 1.055f * pow(color.g, exp) - 0.055f;
  }
  if (color.b < 0.0031308f) {
    color.b = 12.92f * color.b;
  }
  else {
    color.b = 1.055f * pow(color.b, exp) - 0.055f;
  }
}

__device__ float raymarch::clamp(float x, float min, float max) {
  return fminf(fmaxf(x, min), max);
}

__device__ float raymarch::dot2(const glm::vec3& v) {
  return glm::dot(v, v);
}

__device__ glm::vec3 raymarch::getLightDirection(const raymarch::Light& light, const glm::vec3& hitPoint) {
  glm::vec3 direction;
  switch (light.type) {
  case raymarch::LightType::POINT:
    direction = light.pointLight.position - hitPoint;
    break;
  case raymarch::LightType::DIRECTIONAL:
    direction = light.directionalLight.direction;
    break;
  case raymarch::LightType::AREA_PLANE:
    direction = light.areaLightPlane.origin - hitPoint;
    break;
  case raymarch::LightType::AREA_DISC:
    direction = light.areaLightDisc.origin - hitPoint;
    break;
  default:
    printf("Light type not recognized\n");
    direction = glm::vec3(0);
    break;
  }
  return glm::normalize(direction);

}

__device__ void raymarch::perturbRayDirection(raymarch::Ray& ray, curandState* randState, float maxAngle) {
  int curandId = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = randState[curandId];
  float theta = curand_uniform(&localState) * maxAngle;
  float phi = curand_uniform(&localState) * maxAngle;
  glm::vec3 perturb = glm::vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
  ray.direction = glm::normalize(ray.direction + perturb);
  randState[curandId] = localState;
}

__device__ glm::vec2 raymarch::randomInUnitDisk(curandState* state) {
  float r = sqrtf(curand_uniform(state)*0.5f); // Square root for uniform distribution
  float theta = 2.0f * PI * curand_uniform(state); // Full circle

  return glm::vec2(r * cosf(theta), r * sinf(theta));
}

#pragma endregion

#pragma region SHADING FUNCTIONS

__device__ glm::vec3 raymarch::calculatePhongShading(const hitInfo& hit, const Light& light) {
  
  glm::vec3 viewDir = glm::normalize(hit.hitPoint - hit.incidentRay.origin);

  glm::vec3 lightDirection = raymarch::getLightDirection(light, hit.hitPoint);
  glm::vec3 color = raymarch::colorToColorVec( hit.object.color );
  glm::vec3 normal = hit.normal;
  glm::vec3 lightColor = light.color;
  float ambientStrength = 0.1f;
  float shininess = 64.0f;

 //printf("Input Color: %f, %f, %f\n", color.x, color.y, color.z);

  glm::vec3 baseColor = raymarch::colorToColorVec( hit.object.color ) * ambientStrength;


  float diff = glm::dot(normal, lightDirection);
  baseColor += color * fmaxf(diff, 0.0f);

  glm::vec3 reflectDir = glm::reflect(lightDirection, normal);

  float spec = pow(fmaxf(glm::dot(reflectDir, viewDir), 0.0f), shininess);

  baseColor += lightColor * spec;

  

 //printf("Output Color: %f, %f, %f\n", baseColor.x, baseColor.y, baseColor.z);
  return baseColor;

}

__device__ glm::vec3 raymarch::calculateNormalShading(glm::vec3& color, const glm::vec3& normal)
{
  color = (normal + 1.0f) * 0.5f;
}




#pragma endregion

#pragma region SDF FUNCTIONS

__device__ float raymarch::sdf(glm::vec3 checkPoint, raymarch::Renderable* scene, size_t sceneSize, raymarch::Renderable& hitObject) {
  float minDistance = MAX_FLOAT;
  float distance = 0.0f;
  for (int i = 0; i < sceneSize; i++) {
    switch (scene[i].type) {
    case raymarch::PrimitiveType::SPHERE:
      distance = raymarch::sphereSDF(checkPoint, scene[i].sphere);
      break;
    case raymarch::PrimitiveType::BOX:
      distance = raymarch::boxSDF(checkPoint, scene[i].box.halfExtents, scene[i].box.origin);
      break;
    case raymarch::PrimitiveType::BOXFRAME:
      distance = raymarch::boxFrameSDF(checkPoint, scene[i].boxFrame.halfExtents, scene[i].boxFrame.origin, scene[i].boxFrame.edgeWidth);
      break;
    case raymarch::PrimitiveType::ROUNDBOX:
      distance = raymarch::roundBoxSDF(checkPoint, scene[i].roundBox.halfExtents, scene[i].roundBox.origin, scene[i].roundBox.edgeWidth);
      break; 
    case raymarch::PrimitiveType::TRIANGLE:
      distance = raymarch::triangleSDF(checkPoint, scene[i].triangle);
      break;
    case raymarch::PrimitiveType::DISC:
      distance = raymarch::discSDF(checkPoint, scene[i].disc);
      break;
    }
    if (distance < minDistance) {
      hitObject = scene[i];
      minDistance = distance;
    }
  }

  return minDistance;
}

__device__ float raymarch::sphereSDF(glm::vec3 p, const raymarch::Sphere& s) {
  return glm::length(p - s.origin) - s.radius;
}

__device__ float raymarch::boxSDF(glm::vec3 checkPoint, glm::vec3 halfExtents, glm::vec3 origin) {
  glm::vec3 p = checkPoint - origin;
  glm::vec3 d = glm::abs(p) - halfExtents;
  return glm::min(glm::max(d.x, glm::max(d.y, d.z)), 0.0f) + glm::length(glm::max(d, glm::vec3(0.0f)));
}

__device__ float raymarch::boxFrameSDF(glm::vec3 checkPoint, glm::vec3 halfExtents, glm::vec3 origin, float thickness) {
  glm::vec3 p = checkPoint - origin;

  p = abs(p) - halfExtents;
  glm::vec3 q = glm::abs(p + thickness) - thickness;

  return glm::min(glm::min(
    glm::length(glm::max(glm::vec3(p.x, q.y, q.z), 0.0f)) + glm::min(glm::max(p.x, glm::max(q.y, q.z)), 0.0f),
    glm::length(glm::max(glm::vec3(q.x, p.y, q.z), 0.0f)) + glm::min(glm::max(q.x, glm::max(p.y, q.z)), 0.0f)),
    glm::length(glm::max(glm::vec3(q.x, q.y, p.z), 0.0f)) + glm::min(glm::max(q.x, glm::max(q.y, p.z)), 0.0f));
}

__device__ float raymarch::roundBoxSDF(glm::vec3 checkPoint, glm::vec3 halfExtents, glm::vec3 origin, float thickness) {
  glm::vec3 p = checkPoint - origin;
  glm::vec3 d = glm::abs(p) - halfExtents;
  return glm::min(glm::max(d.x, glm::max(d.y, d.z)), 0.0f) + glm::length(glm::max(d, glm::vec3(0.0f))) - thickness;
};

__device__ float raymarch::triangleSDF(glm::vec3 checkPoint, const raymarch::Triangle& t) {
  glm::vec3 ba = t.v1 - t.v0; glm::vec3 pa = checkPoint - t.v0;
  glm::vec3 cb = t.v2 - t.v1; glm::vec3 pb = checkPoint - t.v1;
  glm::vec3 ac = t.v0 - t.v2; glm::vec3 pc = checkPoint - t.v2;
  glm::vec3 nor = glm::cross(ba, ac);

  return sqrtf(
    (
      glm::sign(glm::dot(glm::cross(ba, nor), pa)) +
      glm::sign(glm::dot(glm::cross(cb, nor), pb)) +
      glm::sign(glm::dot(glm::cross(ac, nor), pc)) < 2.0f)
    ?
    fminf(fminf(
      dot2(ba * clamp(glm::dot(ba,pa)/dot2(ba),0.0f,1.0f) - pa),
      dot2(cb * clamp(glm::dot(cb,pb)/dot2(cb),0.0f,1.0f) - pb) ),
      dot2(ac * clamp(glm::dot(ac,pc)/dot2(ac),0.0f,1.0f) - pc) )
    :
    glm::dot(nor, pa) * glm::dot(nor, pa) / glm::dot(nor, nor));

};

__device__ float raymarch::discSDF(glm::vec3 checkPoint, const raymarch::Disc& d) {
  // Vector from checkPoint to the disc's center
  glm::vec3 toCenter = checkPoint - d.origin;

  // Distance from checkPoint to the disc's plane
  float distToPlane = glm::dot(toCenter, d.normal);

  // Project checkPoint onto the disc's plane
  glm::vec3 projPoint = checkPoint - distToPlane * d.normal;

  // Calculate distance from the disc's center to the projection point
  float distOnPlane = glm::length(projPoint - d.origin);

  // Determine if the point is outside the disc's radius on the plane
  float distOutsideRadius = fmaxf(distOnPlane - d.radius, 0.0f);

  // Combine the above distance with the vertical distance to the plane if outside the radius
  float sdf = glm::length(glm::vec2(distOutsideRadius, distToPlane));

  return sdf;
}

__device__ float raymarch::lightSDF(glm::vec3 checkPoint, const Light* light) {
  switch (light->type) {
  case raymarch::LightType::POINT:
    return raymarch::pointLightSDF(checkPoint, light->pointLight);
  case raymarch::LightType::DIRECTIONAL:
    return raymarch::directionalLightSDF(checkPoint, light->directionalLight);
  case raymarch::LightType::AREA_PLANE:
    return raymarch::areaLightSDF(checkPoint, light->areaLightPlane);
  case raymarch::LightType::AREA_DISC:
    return raymarch::discLightSDF(checkPoint, light->areaLightDisc);
  default:
   //printf("Light type not recognized\n");
    return 0.0f;
  }

}

__device__ float raymarch::pointLightSDF(glm::vec3 checkPoint, const PointLight& light) {
  return glm::length(checkPoint - light.position) - light.radius;

}

__device__ float raymarch::directionalLightSDF(glm::vec3 checkPoint, const DirectionalLight& light) {
  return 0.0f;
}

// TODO __device__ float raymarch::spotLightSDF(glm::vec3 checkPoint, const SpotLight& light);
// TODO __device__ float raymarch::sphereLightSDF(glm::vec3 checkPoint, const SphereLight& light);
__device__ float raymarch::areaLightSDF(glm::vec3 checkPoint, const AreaLightPlane& light) {
  
  //the area light plane is defined with an origin, normal, and two vectors that define the plane
  //we can use the two vectors to define two triangles and then use the triangle SDF to get the distance
  
  raymarch::Triangle t1;
  t1.v0 = light.origin + light.u + light.v;
  t1.v1 = light.origin + light.u;
  t1.v2 = light.origin;

  raymarch::Triangle t2;
  t2.v0 = light.origin + light.v;
  t2.v1 = light.origin + light.u + light.v;
  t2.v2 = light.origin;

  //debug print t1 and t2 points
  //printf("t1.v0: %f, %f, %f\n", t1.v0.x, t1.v0.y, t1.v0.z);
  //printf("t1.v1: %f, %f, %f\n", t1.v1.x, t1.v1.y, t1.v1.z);
  //printf("t1.v2: %f, %f, %f\n", t1.v2.x, t1.v2.y, t1.v2.z);
  //printf("t2.v0: %f, %f, %f\n", t2.v0.x, t2.v0.y, t2.v0.z);
  //printf("t2.v1: %f, %f, %f\n", t2.v1.x, t2.v1.y, t2.v1.z);
  //printf("t2.v2: %f, %f, %f\n", t2.v2.x, t2.v2.y, t2.v2.z);


  return fminf(raymarch::triangleSDF(checkPoint, t1), raymarch::triangleSDF(checkPoint, t2));

}

__device__ float raymarch::discLightSDF(glm::vec3 checkPoint, const AreaLightDisc& light) {
  
  raymarch::Disc disc;
  disc.origin = light.origin;
  disc.normal = light.normal;
  disc.radius = light.radius;
  return raymarch::discSDF(checkPoint, disc);

}

#pragma endregion

#pragma region NORMALS FUNCTIONS

__device__ glm::vec3 raymarch::getNormal(const raymarch::Renderable& hitObject, glm::vec3 p) {
  switch (hitObject.type) {
  case raymarch::PrimitiveType::SPHERE:
    return glm::normalize(p - hitObject.sphere.origin);
  case raymarch::PrimitiveType::BOX:
    return raymarch::boxNormal(p, hitObject.box.halfExtents, hitObject.box.origin);
  case raymarch::PrimitiveType::BOXFRAME:
    return raymarch::boxFrameNormal(p, hitObject.boxFrame.halfExtents, hitObject.boxFrame.origin, hitObject.boxFrame.edgeWidth);
  case raymarch::PrimitiveType::ROUNDBOX:
    return raymarch::roundBoxNormal(p, hitObject.roundBox.halfExtents, hitObject.roundBox.origin, hitObject.roundBox.edgeWidth);
  case raymarch::PrimitiveType::TRIANGLE:
    return raymarch::triangleNormal(p, hitObject.triangle);
  case raymarch::PrimitiveType::DISC:
    return raymarch::discNormal(p, hitObject.disc);
  }
  return glm::vec3(0, 0, 0);
}

__device__ glm::vec3 raymarch::boxNormal(glm::vec3 hitPoint, glm::vec3 halfExtents, glm::vec3 origin)
{
  glm::vec3 normal;

  glm::vec3 p = hitPoint - origin;
  glm::vec3 q = glm::abs(p) - halfExtents;
  float maxComp = glm::max(q.x, glm::max(q.y, q.z));

  normal.x = float(q.x == maxComp) * glm::sign(p.x);
  normal.y = float(q.y == maxComp) * glm::sign(p.y);
  normal.z = float(q.z == maxComp) * glm::sign(p.z);


  return glm::normalize(normal);
}

__device__ glm::vec3 raymarch::boxFrameNormal(glm::vec3 hitPoint, glm::vec3 halfExtents, glm::vec3 origin, float thickness)
{
  glm::vec3 normal;

  float eps = 0.0001f;
  normal.x = raymarch::boxFrameSDF(hitPoint + glm::vec3(eps, 0, 0), halfExtents, origin, thickness) - raymarch::boxFrameSDF(hitPoint - glm::vec3(eps, 0, 0), halfExtents, origin, thickness);
  normal.y = raymarch::boxFrameSDF(hitPoint + glm::vec3(0, eps, 0), halfExtents, origin, thickness) - raymarch::boxFrameSDF(hitPoint - glm::vec3(0, eps, 0), halfExtents, origin, thickness);
  normal.z = raymarch::boxFrameSDF(hitPoint + glm::vec3(0, 0, eps), halfExtents, origin, thickness) - raymarch::boxFrameSDF(hitPoint - glm::vec3(0, 0, eps), halfExtents, origin, thickness);

  return glm::normalize(normal);
}

__device__ glm::vec3 raymarch::roundBoxNormal(glm::vec3 hitPoint, glm::vec3 halfExtents, glm::vec3 origin, float thickness)
{
  glm::vec3 normal;

  float eps = 0.0001f;
  normal.x = raymarch::roundBoxSDF(hitPoint + glm::vec3(eps, 0, 0), halfExtents, origin, thickness) - raymarch::roundBoxSDF(hitPoint - glm::vec3(eps, 0, 0), halfExtents, origin, thickness);
  normal.y = raymarch::roundBoxSDF(hitPoint + glm::vec3(0, eps, 0), halfExtents, origin, thickness) - raymarch::roundBoxSDF(hitPoint - glm::vec3(0, eps, 0), halfExtents, origin, thickness);
  normal.z = raymarch::roundBoxSDF(hitPoint + glm::vec3(0, 0, eps), halfExtents, origin, thickness) - raymarch::roundBoxSDF(hitPoint - glm::vec3(0, 0, eps), halfExtents, origin, thickness);

  return glm::normalize(normal);
}

__device__ glm::vec3 raymarch::estimateNormal(glm::vec3 p, const  raymarch::Renderable& hitObject)
{
  float eps = 0.0001f;
  glm::vec3 normal;
  switch (hitObject.type) {
  case raymarch::PrimitiveType::SPHERE:
    return glm::normalize(p - hitObject.sphere.origin);
  case raymarch::PrimitiveType::BOX:
    normal.x = raymarch::boxSDF(p + glm::vec3(eps, 0, 0), hitObject.box.halfExtents, hitObject.box.origin) - raymarch::boxSDF(p - glm::vec3(eps, 0, 0), hitObject.box.halfExtents, hitObject.box.origin);
    normal.y = raymarch::boxSDF(p + glm::vec3(0, eps, 0), hitObject.box.halfExtents, hitObject.box.origin) - raymarch::boxSDF(p - glm::vec3(0, eps, 0), hitObject.box.halfExtents, hitObject.box.origin);
    normal.z = raymarch::boxSDF(p + glm::vec3(0, 0, eps), hitObject.box.halfExtents, hitObject.box.origin) - raymarch::boxSDF(p - glm::vec3(0, 0, eps), hitObject.box.halfExtents, hitObject.box.origin);
    break;
  }
  return glm::normalize(normal);
}

__device__ glm::vec3 raymarch::triangleNormal(glm::vec3 hitPoint, const raymarch::Triangle& t) {
  glm::vec3 normal = glm::cross(t.v1 - t.v0, t.v2 - t.v0);
  return glm::normalize(normal);
}

__device__ glm::vec3 raymarch::discNormal(glm::vec3 hitPoint, const raymarch::Disc& d) {
  return d.normal;
}

#pragma endregion