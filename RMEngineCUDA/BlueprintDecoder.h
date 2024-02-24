#pragma once
#include "Renderable.h"
#include "Camera.cuh"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

struct SceneSettings
{
  int width = 800;
  int height = 600;
  bool enableAntiAliasing = false;
  int antiAliasingQuality = 1;
  bool enableDoF = false;
};

class BlueprintDecoder
{
private:
  bool debug;
  std::string filepath;
  std::vector<raymarch::Renderable> renderables;
  std::vector<raymarch::Light> lights;
  Camera camera;
  bool parseBlueprint();
  SceneSettings settings;
  void convertToBool(std::istringstream& iss, bool& b);
  bool parseCamera(std::ifstream& file);
  bool parseRenderables(std::ifstream& file);
  bool parseLights(std::ifstream& file);
  bool parseObjectOfType(const std::string& type, std::ifstream& file, bool& eos);
  bool parseLightOfType(const std::string& type, std::ifstream& file, bool& eos);

  bool parseSphere(std::ifstream& file, raymarch::Sphere& sphere, bool& eos);
  bool parseTriangle(std::ifstream& file, raymarch::Triangle& tri, bool& eos);
  bool parseBox(std::ifstream& file, raymarch::Box& box, bool& eos);
  bool parseDisk(std::ifstream& file, raymarch::Disc& disc, bool& eos);
  bool parseRoundBox(std::ifstream& file, raymarch::RoundBox& rBox, bool& eos);
  bool parseBoxFrame(std::ifstream& file, raymarch::BoxFrame& bf, bool& eos);

  bool parseDirectionalLight(std::ifstream& file, raymarch::DirectionalLight& dl, bool& eos);
  
  template <typename T>
  bool allValuesInMapAreTrue(std::map<T, bool>& m);

  void printDebug(const std::string& s);

  bool checkLineIsEos(const std::string& line);

public:
  BlueprintDecoder(const std::string& fp, bool debug);
  ~BlueprintDecoder() {}
  std::vector<raymarch::Renderable> getRenderables() { return renderables; };
  std::vector<raymarch::Light> getLights() const { return lights; };
  Camera getCamera() const { return camera; };
  SceneSettings getSettings() const { return settings; };
};


namespace util {
  std::string ltrim(const std::string& s);
  std::string rtrim(const std::string& s);
  std::string trim(const std::string& s);
}