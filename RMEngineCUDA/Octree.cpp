#include "Octree.h"
#include <vector>
#include <cstdio>
#include <stdexcept>


raymarch::Box Octree::getSceneAABB()
{
  glm::vec3 min = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
  glm::vec3 max = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  raymarch::Box aabb;

  for (const raymarch::Renderable& r : scene) {
    switch (r.type) {
    case raymarch::PrimitiveType::SPHERE:
      min = glm::min(min, r.sphere.origin - glm::vec3(r.sphere.radius));
      max = glm::max(max, r.sphere.origin + glm::vec3(r.sphere.radius));
      break;
    case raymarch::PrimitiveType::BOXFRAME:
      min = glm::min(min, r.boxFrame.origin - r.boxFrame.halfExtents);
      max = glm::max(max, r.boxFrame.origin + r.boxFrame.halfExtents);
      break;
    case raymarch::PrimitiveType::TRIANGLE:
      min = glm::min(min, r.triangle.v0);
      min = glm::min(min, r.triangle.v1);
      min = glm::min(min, r.triangle.v2);
      max = glm::max(max, r.triangle.v0);
      max = glm::max(max, r.triangle.v1);
      max = glm::max(max, r.triangle.v2);
      break;
    case raymarch::PrimitiveType::BOX:
      min = glm::min(min, r.box.origin);
      max = glm::max(max, r.box.origin + r.box.halfExtents);
      break;
    case raymarch::PrimitiveType::ROUNDBOX:
      min = glm::min(min, r.roundBox.origin - r.roundBox.halfExtents);
      max = glm::max(max, r.roundBox.origin + r.roundBox.halfExtents);
      break;
    }
  }

  aabb.halfExtents = (max - min) / 2.0f;
  aabb.origin = (max + min) / 2.0f;

  return aabb;
}

void Octree::build() {
  
  raymarch::Box primaryAABB = getSceneAABB();
  OctreeNode* root = new OctreeNode();
  root->parentIdx = -1;
  root->nodeIdx = 0;
  root->bounds = primaryAABB;
  root->containsData = true;

  root->bounds.halfExtents;

  for (int i = 0; i < 8; i++) {
    root->childIndices[i] = -1;
  }

  ptree.push_back(root);
  splitBounds(root, 0);
  transferPtreeToTree();
  //if (!validateTree()) {
  //  throw std::runtime_error("Invalid Octree");
  //}

}

void Octree::splitBounds(OctreeNode* root, int recusionDepth)
{
  if (recusionDepth >= MAX_OCTREE_DEPTH) {
    return;
  }

  float epsilonOverlap = 0.55f;
  glm::vec3 offsets[8] = {
    {-1, -1, -1},
    { 1, -1, -1},
    {-1,  1, -1},
    { 1,  1, -1},
    {-1, -1,  1},
    { 1, -1,  1},
    {-1,  1,  1},
    { 1,  1,  1}
  };
  // split the bounds into 8 sub bounds, each sub bound is child of the root.  push back the child bounds into the octree
  glm::vec3 halfExtents = root->bounds.halfExtents * 0.5f;
  raymarch::Box subBounds[8];
  OctreeNode* subNodes[8];

  for (int i = 0; i < 8; i++) {
    subBounds[i].halfExtents = halfExtents + epsilonOverlap;
    subBounds[i].origin = root->bounds.origin + halfExtents * offsets[i];

    subNodes[i] = new OctreeNode();
    subNodes[i]->bounds = subBounds[i];
    subNodes[i]->containsData = boxContainsObject(subBounds[i]);

    for (int j = 0; j < 8; j++) {
      subNodes[i]->childIndices[j] = -1;
    }

    if (subNodes[i]->containsData) {
      root->childIndices[i] = ptree.size();
      subNodes[i]->nodeIdx = ptree.size();
      subNodes[i]->parentIdx = root->nodeIdx;
      ptree.push_back(subNodes[i]);
      splitBounds(subNodes[i], recusionDepth + 1);
    }

  }

}

bool Octree::boxContainsObject(const raymarch::Box& box) const
{
  for (const raymarch::Renderable& r : scene) {
    raymarch::Box aabb = getObjAABB(r);
    // Check for non-overlap and continue if true
    if (aabb.origin.x - aabb.halfExtents.x > box.origin.x + box.halfExtents.x ||
      aabb.origin.x + aabb.halfExtents.x < box.origin.x - box.halfExtents.x ||
      aabb.origin.y - aabb.halfExtents.y > box.origin.y + box.halfExtents.y ||
      aabb.origin.y + aabb.halfExtents.y < box.origin.y - box.halfExtents.y ||
      aabb.origin.z - aabb.halfExtents.z > box.origin.z + box.halfExtents.z ||
      aabb.origin.z + aabb.halfExtents.z < box.origin.z - box.halfExtents.z) {
      continue; // No overlap, check the next object
    }
    // Overlap detected, the box contains this object
    return true;
  }
  // No objects are contained within the box
  return false;
}

raymarch::Box Octree::getObjAABB(const raymarch::Renderable& obj) const {
  raymarch::Box aabb;
  switch (obj.type) {
  case raymarch::PrimitiveType::SPHERE:
    aabb.halfExtents = glm::vec3(obj.sphere.radius);
    aabb.origin = obj.sphere.origin;
    break;
  case raymarch::PrimitiveType::BOXFRAME:
    aabb.halfExtents = obj.boxFrame.halfExtents;
    aabb.origin = obj.boxFrame.origin;
    break;
  case raymarch::PrimitiveType::TRIANGLE:
    glm::vec3 min = glm::min(obj.triangle.v0, glm::min(obj.triangle.v1, obj.triangle.v2));
    glm::vec3 max = glm::max(obj.triangle.v0, glm::max(obj.triangle.v1, obj.triangle.v2));
    aabb.origin = (max + min) / 2.0f;
    aabb.halfExtents = (max - min) / 2.0f;
    break;
  case raymarch::PrimitiveType::BOX:
    aabb.halfExtents = obj.box.halfExtents;
    aabb.origin = obj.box.origin;
    break;
  case raymarch::PrimitiveType::ROUNDBOX:
    aabb.halfExtents = obj.roundBox.halfExtents;
    aabb.origin = obj.roundBox.origin;
    break;
  case raymarch::PrimitiveType::DISC:
    aabb.halfExtents = glm::vec3(obj.disc.radius);
    aabb.origin = obj.disc.origin;
    break;
  }
  return aabb;
}

void Octree::debugDump() {
  recursiveDump(tree[0], 0, 0);
}

void Octree::recursiveDump(const OctreeNode& node, int nodeIdx, int depth) {
  printf("Node %d\n", nodeIdx);
  printf("Depth %d\n", depth);
  printf("Bounds: (%f, %f, %f) (%f, %f, %f)\n", node.bounds.origin.x, node.bounds.origin.y, node.bounds.origin.z, node.bounds.halfExtents.x, node.bounds.halfExtents.y, node.bounds.halfExtents.z);
  printf("Contains Data: %d\n", node.containsData);
  printf("Children: ");
  for (int j = 0; j < 8; j++) {
    printf("%d ", node.childIndices[j]);
  }
  printf("\n");
  for (int i = 0; i < 8; i++) {
    if (node.childIndices[i] != -1) {
      recursiveDump(tree[node.childIndices[i]], node.childIndices[i], depth + 1);
    }
  }

}

bool Octree::validateTree()
{
  OctreeNode root = tree[0];
  for (int i = 0; i < 8; i++) {
    if (root.childIndices[i] != -1) {
      if (!recurisveValidateTree(tree[root.childIndices[i]], root)) {
        return false;
      }
    }
  }
  return true;
}

bool Octree::recurisveValidateTree(const OctreeNode& node, const OctreeNode& parent)
{
  if (!validateNodeIsContainedByParent(node, parent)) {
    return false;
  }
  for (int i = 0; i < 8; i++) {
    if (node.childIndices[i] != -1) {
      if (!recurisveValidateTree(tree[node.childIndices[i]], node)) {
        return false;
      }
    }
  }
  return true;
}

bool Octree::validateNodeIsContainedByParent(const OctreeNode& node, const OctreeNode& parent) {
  const float epsilon = 1e-6;
  glm::vec3 min = parent.bounds.origin - parent.bounds.halfExtents;
  glm::vec3 max = parent.bounds.origin + parent.bounds.halfExtents;
  glm::vec3 nodeMin = node.bounds.origin - node.bounds.halfExtents;
  glm::vec3 nodeMax = node.bounds.origin + node.bounds.halfExtents;
  if (nodeMin.x < min.x - epsilon || nodeMin.y < min.y - epsilon || nodeMin.z < min.z - epsilon) {
    return false;
  }
  if (nodeMax.x > max.x + epsilon || nodeMax.y > max.y + epsilon || nodeMax.z > max.z + epsilon) {
    return false;
  }
  return true;

}


void Octree::transferPtreeToTree()
{
  for (int i = 0; i < ptree.size(); i++) {
    tree.push_back(*ptree[i]);
  }
}

Octree::~Octree()
{
  for (int i = 0; i < ptree.size(); i++) {
    delete ptree[i];
  }
}