#ifndef OCTREE_H
#define OCTREE_H


#include <vector>
#include "renderable.h"


#define MAX_OCTREE_DEPTH 2

struct OctreeNode {
  raymarch::Box bounds;
  bool containsData;
  int parentIdx;
  int nodeIdx;
  int childIndices[8];
};

class Octree {
private:
  void build();
  raymarch::Box getSceneAABB();
  void splitBounds(OctreeNode* root, int recusionDepth);
  bool boxContainsObject(const raymarch::Box& box) const;
  std::vector<OctreeNode*> ptree;
  raymarch::Box getObjAABB(const raymarch::Renderable& obj) const;
  void transferPtreeToTree();
  void recursiveDump(const OctreeNode& node, int nodeIdx, int depth);

  bool validateTree();
  bool recurisveValidateTree(const OctreeNode& node, const OctreeNode& parent);
  bool validateNodeIsContainedByParent(const OctreeNode& node, const OctreeNode& parent);

public:
  Octree(const std::vector<raymarch::Renderable>& s) : scene(s) { build(); };
  ~Octree();
  std::vector<OctreeNode> tree;
  const std::vector<raymarch::Renderable>& scene;
  OctreeNode* data() { return tree.data(); };
  size_t size() { return tree.size(); };
  void debugDump();

};


#endif