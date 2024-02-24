#include "BlueprintDecoder.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <map>

#include <glm/glm.hpp> 

std::string ERROR_STRING = "";

std::map<std::string, raymarch::PrimitiveType> PRIMITIVE_TYPE_MAP = {
  {"SPHERE", raymarch::PrimitiveType::SPHERE},
  {"TRIANGLE", raymarch::PrimitiveType::TRIANGLE},
  {"BOX", raymarch::PrimitiveType::BOX},
  {"DISC", raymarch::PrimitiveType::DISC},
  {"ROUNDBOX", raymarch::PrimitiveType::ROUNDBOX},
  {"BOXFRAME", raymarch::PrimitiveType::BOXFRAME}
};

std::map<std::string, raymarch::LightType> LIGHT_TYPE_MAP = {
  {"DIRECTIONAL", raymarch::LightType::DIRECTIONAL},
  {"POINT", raymarch::LightType::POINT},
};

BlueprintDecoder::BlueprintDecoder(const std::string& fp, bool debug)
{
  filepath = fp;
  this->debug = debug;
  if (!parseBlueprint()) {
    throw std::runtime_error("Error parsing blueprint. Missing value: " + ERROR_STRING);
  }
}

void BlueprintDecoder::convertToBool(std::istringstream& iss, bool& value)
{
  int val;
  iss >> val;
  value = val == 1;
}

bool BlueprintDecoder::parseBlueprint()
{
  printDebug("Parsing blueprint...");
  std::ifstream file(filepath);
  std::string line;
  
  std::map<std::string, bool> requiredSettingsFound = {
    {"WIDTH", false},
    {"HEIGHT", false},
    {"ENABLE_ANTIALIASING", false},
    {"ANTIALIASING_QUALITY", false},
    {"WITH_DOF", false},
  };

  while (std::getline(file, line))
  {
    std::istringstream iss(line);
    std::string keyword;
    iss >> keyword;

    keyword = util::trim(keyword);

    if (keyword != "") {
      printDebug("Current keyword: " + keyword);
    }

    if (keyword == "WIDTH") {
      printDebug("Parsing width...");
      iss >> settings.width;
      requiredSettingsFound["WIDTH"] = true;
    }
    else if (keyword == "HEIGHT") {
      printDebug("Parsing height...");
      iss >> settings.height;
      requiredSettingsFound["HEIGHT"] = true;
    }
    else if (keyword == "ENABLE_ANTIALIASING") {
      printDebug("Parsing antialiasing...");
      convertToBool(iss, settings.enableAntiAliasing);
      requiredSettingsFound["ENABLE_ANTIALIASING"] = true;
    }
    else if (keyword == "ANTIALIASING_QUALITY") {
      printDebug("Parsing antialiasing quality...");
      requiredSettingsFound["ANTIALIASING_QUALITY"] = true;
      iss >> settings.antiAliasingQuality;
    }
    else if (keyword == "WITH_DOF") {
      printDebug("Parsing DOF...");
      convertToBool(iss, settings.enableDoF);
      requiredSettingsFound["WITH_DOF"] = true;
    }
    else if (keyword == "CAMERA") {
      printDebug("Parsing camera...");
      if (!parseCamera(file)) return false;
    }
    else if (keyword == "OBJECT") {
      printDebug("Parsing renderables...");
      if (!parseRenderables(file)) return false;
    }
    else if (keyword == "LIGHT") {
      printDebug("Parsing lights...");
      if (!parseLights(file)) return false;
    }

  }
  return allValuesInMapAreTrue(requiredSettingsFound);
}

bool BlueprintDecoder::parseCamera(std::ifstream& file)
{
  std::string line;
  std::string keyword;
  std::map<std::string, bool> requiredSettingsFound = {
    {"POSITION", false},
    {"LOOK_AT", false},
    {"UP", false},
    {"FOV", false},
    {"APERTURE", false},
    {"FOCAL_DISTANCE", false}
  };

  while (std::getline(file, line) && line != "}") {
    std::istringstream iss(line);
    iss >> keyword;
    keyword = util::trim(keyword);
    if (keyword != "") {
      printDebug("Current keyword: " + keyword);
    }
    if (keyword == "POSITION") {
      printDebug("Parsing camera position...");
      iss >> camera.position.x >> camera.position.y >> camera.position.z;
      requiredSettingsFound["POSITION"] = true;
    }
    else if (keyword == "LOOK_AT") {
      printDebug("Parsing camera look at...");
      iss >> camera.lookAt.x >> camera.lookAt.y >> camera.lookAt.z;
      requiredSettingsFound["LOOK_AT"] = true;
    }
    else if (keyword == "UP") {
      printDebug("Parsing camera up...");
      iss >> camera.up.x >> camera.up.y >> camera.up.z;
      requiredSettingsFound["UP"] = true;
    }
    else if (keyword == "FOV") {
      printDebug("Parsing camera fov...");
      iss >> camera.fov;
      requiredSettingsFound["FOV"] = true;
    }
    else if (keyword == "APERTURE") {
      printDebug("Parsing camera aperture...");
      iss >> camera.aperture;
      requiredSettingsFound["APERTURE"] = true;
    }
    else if (keyword == "FOCAL_DISTANCE") {
      printDebug("Parsing camera focal distance...");
      iss >> camera.focalLength;
      requiredSettingsFound["FOCAL_DISTANCE"] = true;
    }
  }
  return allValuesInMapAreTrue(requiredSettingsFound);
}

bool BlueprintDecoder::parseLights(std::ifstream& file)
{
  std::string line;
  std::string keyword;
  std::map<std::string, bool> requiredSettingsFound = {
    {"TYPE", false},
    {"COLOR", false},
  };
  glm::vec3 color = glm::vec3(0);

  bool eos = false;

  while (std::getline(file, line) && !eos) {
    

    std::istringstream iss(line);
    iss >> keyword;
    keyword = util::trim(keyword);
    if (keyword != "") {
      printDebug("Current keyword: " + keyword);
    }

    if (keyword == "{")
    {
      continue;
    }
    else if (keyword == "}") {
      eos = true;
      break;
    }
    else if (keyword == "COLOR") {
      printDebug("Parsing light color...");
      requiredSettingsFound["COLOR"] = true;
      iss >> color.r >> color.g >> color.b;
    }
    else if (keyword == "TYPE") {
      printDebug("Parsing light type...");
      std::string type;
      iss >> type;
      requiredSettingsFound["TYPE"] = true;
      if (!parseLightOfType(type, file, eos)) return false;
      lights.back().color = color;
    }

  }
  return allValuesInMapAreTrue(requiredSettingsFound);
}

bool BlueprintDecoder::parseRenderables(std::ifstream& file)
{
  printDebug("Parsing renderables...");
  std::string line;
  std::string keyword;
  raymarch::Color color = { 0, 0, 0, 255 };
  glm::vec3 sColor = {0.0f, 0.0f, 0.0f};
  float ambient = 0.0f;

  std::map<std::string, bool> requiredSettingsFound = {
   {"TYPE", false},
   {"COLOR", false}
  };
  bool eos = false;
  while (std::getline(file, line) && !eos) {
    std::istringstream iss(line);
    iss >> keyword;
    keyword = util::trim(keyword);
    if (keyword != "") {
      printDebug("Current keyword: " + keyword);
    }

    if (keyword == "{")
    {
      continue;
    }
    else if (keyword == "}") {
      eos = true;
      break;
    }
    else if (keyword == "COLOR") {
      printDebug("Parsing renderable color...");
      requiredSettingsFound["COLOR"] = true;
      iss >> sColor.r >> sColor.g >> sColor.b >> ambient;
      color = {(unsigned char)sColor.r, (unsigned char)sColor.g, (unsigned char)sColor.b, (unsigned char)ambient };
    }
    else if (keyword == "TYPE") {
      printDebug("Parsing renderable type...");
      std::string type;
      iss >> type;
      requiredSettingsFound["TYPE"] = true;
      if (!parseObjectOfType(type, file, eos)) return false;
      renderables.back().color = color;
    }
  }
  return allValuesInMapAreTrue(requiredSettingsFound);
}


bool BlueprintDecoder::parseObjectOfType(const std::string& iType, std::ifstream& file, bool& eos)
{
  std::string type = iType;
  type = util::trim(type);

  if (type.empty()) {
    ERROR_STRING = "Object Type cannot be empty";
    return false;
  }

  printDebug("Parsing object of type: " + type);
  bool isValid = false;
  raymarch::Renderable renderable;
  renderable.type = PRIMITIVE_TYPE_MAP[type];
  if (type == "SPHERE") {
    isValid = parseSphere(file, renderable.sphere, eos);
  }
  else if (type == "TRIANGLE") {
    isValid = parseTriangle(file, renderable.triangle, eos);
  }
  else if (type == "BOX") {
    isValid = parseBox(file, renderable.box, eos);
  }
  else if (type == "DISC") {
    isValid = parseDisk(file, renderable.disc, eos);
  }
  else if (type == "ROUNDBOX") {
    isValid = parseRoundBox(file, renderable.roundBox, eos);
  }
  else if (type == "BOXFRAME") {
    isValid = parseBoxFrame(file, renderable.boxFrame, eos);
  }

  if (isValid) {
    renderables.push_back(renderable);
  }
  printDebug("isValid: " + std::to_string(isValid));
  return isValid;
}

bool BlueprintDecoder::parseLightOfType(const std::string& iType, std::ifstream& file, bool& eos)
{
  std::string type = iType;
  type = util::trim(type);

if (type.empty()) {
    ERROR_STRING = "Light Type cannot be empty";
    return false;
  }

  printDebug("Parsing light of type: " + type);
  bool isValid = false;
  raymarch::Light light;
  light.type = LIGHT_TYPE_MAP[type];
  if (type == "DIRECTIONAL") {
    isValid = parseDirectionalLight(file, light.directionalLight, eos);
  }
  //else if (type == "POINT") {
  //  isValid = parsePointLight(file, light);
  //}
  //else if (type == "SPOT") {
  //  isValid = parseSpotLight(file, light);
  //}
  if (isValid) {
    lights.push_back(light);
  }
  printDebug("isValid: " + std::to_string(isValid));
  return isValid;
}

#pragma region ParseRenderables

bool BlueprintDecoder::parseSphere(std::ifstream& file, raymarch::Sphere& sphere, bool& eos)
{
  std::string line;
  std::string keyword;
  std::map<std::string, bool> requiredSettingsFound = {
    {"POSITION", false},
    {"RADIUS", false}
  };

  

  while (std::getline(file, line) && !eos) {
    std::istringstream iss(line);
    if(checkLineIsEos(line)) {
      eos = true;
      break;
    }
    iss >> keyword;
    keyword = util::trim(keyword);
    if (keyword == "POSITION") {
      iss >> sphere.origin.x >> sphere.origin.y >> sphere.origin.z;
      requiredSettingsFound["POSITION"] = true;
    }
    else if (keyword == "RADIUS") {
      iss >> sphere.radius;
      requiredSettingsFound["RADIUS"] = true;
    }
  }

  return allValuesInMapAreTrue(requiredSettingsFound);
}

bool BlueprintDecoder::parseTriangle(std::ifstream& file, raymarch::Triangle& triangle, bool& eos)
{
  std::string line;
  std::string keyword;
  std::map<std::string, bool> requiredSettingsFound = {
    {"V0", false},
    {"V1", false},
    {"V2", false}
  };

  while (std::getline(file, line)) {
    if (checkLineIsEos(line)) {
      eos = true;
      break;
    }
    std::istringstream iss(line);
    iss >> keyword;
    keyword = util::trim(keyword);
    if (keyword == "V0") {
      iss >> triangle.v0.x >> triangle.v0.y >> triangle.v0.z;
      requiredSettingsFound["V0"] = true;
    }
    else if (keyword == "V1") {
      iss >> triangle.v1.x >> triangle.v1.y >> triangle.v1.z;
      requiredSettingsFound["V1"] = true;
    }
    else if (keyword == "V2") {
      iss >> triangle.v2.x >> triangle.v2.y >> triangle.v2.z;
      requiredSettingsFound["V2"] = true;
    }
  }
  return allValuesInMapAreTrue(requiredSettingsFound);
}

bool BlueprintDecoder::parseBox(std::ifstream& file, raymarch::Box& box, bool& eos)
{
  std::string line;
  std::string keyword;
  std::map<std::string, bool> requiredSettingsFound = {
    {"POSITION", false},
    {"HALF_EXTENTS", false}
  };

  while (std::getline(file, line)) {
    if (checkLineIsEos(line)) {
      eos = true;
      break;
    }
    std::istringstream iss(line);
    iss >> keyword;
    keyword = util::trim(keyword);
    if (keyword == "POSITION") {
      iss >> box.origin.x >> box.origin.y >> box.origin.z;
      requiredSettingsFound["MIN"] = true;
    }
    else if (keyword == "MAX") {
      iss >> box.halfExtents.x >> box.halfExtents.y >> box.halfExtents.z;
      requiredSettingsFound["HALF_EXTENTS"] = true;
    }
  }
  return allValuesInMapAreTrue(requiredSettingsFound);
}

bool BlueprintDecoder::parseDisk(std::ifstream& file, raymarch::Disc& disc, bool& eos) {
  std::string line;
  std::string keyword;

  std::map<std::string, bool> requiredSettingsFound = {
    {"POSITION", false},
    {"NORMAL", false},
    {"RADIUS", false}
  };

  while (std::getline(file, line)) {
    if (checkLineIsEos(line)) {
      eos = true;
      break;
    }
    std::istringstream iss(line);
    iss >> keyword;
    keyword = util::trim(keyword);
    if (keyword == "POSITION") {
      iss >> disc.origin.x >> disc.origin.y >> disc.origin.z;
      requiredSettingsFound["POSITION"] = true;
    }
    else if (keyword == "NORMAL") {
      iss >> disc.normal.x >> disc.normal.y >> disc.normal.z;
      requiredSettingsFound["NORMAL"] = true;
    }
    else if (keyword == "RADIUS") {
      iss >> disc.radius;
      requiredSettingsFound["RADIUS"] = true;
    }
  }
  return allValuesInMapAreTrue(requiredSettingsFound);
}

bool BlueprintDecoder::parseRoundBox(std::ifstream& file, raymarch::RoundBox& box, bool& eos) {
  std::string line;
  std::string keyword;
  std::map<std::string, bool> requiredSettingsFound = {
    {"POSITION", false},
    {"HALF_EXTENTS", false},
    {"RADIUS", false}
  };

  while (std::getline(file, line)) {
    if (checkLineIsEos(line)) {
      eos = true;
      break;
    }
    std::istringstream iss(line);
    iss >> keyword;
    keyword = util::trim(keyword);
    if (keyword == "POSITION") {
      iss >> box.origin.x >> box.origin.y >> box.origin.z;
      requiredSettingsFound["POSITION"] = true;
    }
    else if (keyword == "HALF_EXTENTS") {
      iss >> box.halfExtents.x >> box.halfExtents.y >> box.halfExtents.z;
      requiredSettingsFound["HALF_EXTENTS"] = true;
    }
    else if (keyword == "RADIUS") {
      iss >> box.edgeWidth;
      requiredSettingsFound["RADIUS"] = true;
    }
  }
  return allValuesInMapAreTrue(requiredSettingsFound);
}

bool BlueprintDecoder::parseBoxFrame(std::ifstream& file, raymarch::BoxFrame& bf, bool& eos) {
  std::string line;
  std::string keyword;
  std::map<std::string, bool> requiredSettingsFound = {
    {"POSITION", false},
    {"HALF_EXTENTS", false},
    {"EDGE_WIDTH", false}
  };

  while (std::getline(file, line)) {
    if (checkLineIsEos(line)) {
      eos = true;
      break;
    }
    std::istringstream iss(line);
    iss >> keyword;
    keyword = util::trim(keyword);
    if (keyword == "POSITION") {
      iss >> bf.origin.x >> bf.origin.y >> bf.origin.z;
      requiredSettingsFound["POSITION"] = true;
    }
    else if (keyword == "HALF_EXTENTS") {
      iss >> bf.halfExtents.x >> bf.halfExtents.y >> bf.halfExtents.z;
      requiredSettingsFound["HALF_EXTENTS"] = true;
    }
    else if (keyword == "EDGE_WIDTH") {
      iss >> bf.edgeWidth;
      requiredSettingsFound["EDGE_WIDTH"] = true;
    }
  }
  return allValuesInMapAreTrue(requiredSettingsFound);
}

#pragma endregion

#pragma region ParseLights

bool BlueprintDecoder::parseDirectionalLight(std::ifstream& file, raymarch::DirectionalLight& light, bool& eos)
{
  std::string line;
  std::string keyword;
  std::map<std::string, bool> requiredSettingsFound = {
    {"DIRECTION", false}
  };
  
  while (std::getline(file, line)) {
    if (checkLineIsEos(line)) {
      eos = true;
      break;
    }

    std::istringstream iss(line);
    iss >> keyword;
    keyword = util::trim(keyword);
    if (keyword == "DIRECTION") {
      printDebug("Parsing light direction...");
      iss >> light.direction.x >> light.direction.y >> light.direction.z;
      requiredSettingsFound["DIRECTION"] = true;
    }
  }
  return allValuesInMapAreTrue(requiredSettingsFound);
}


#pragma endregion

#pragma region utility

template <typename T>
bool BlueprintDecoder::allValuesInMapAreTrue(std::map<T, bool>& map)
{
  printDebug("Checking if all values in map are true...");
  for (auto& pair : map) {
    printDebug(pair.first + " " + std::to_string(pair.second));
    if (!pair.second) {
      ERROR_STRING = pair.first;
      return false;
    }
  }
  return true;
}


void BlueprintDecoder::printDebug(const std::string& msg)
{
  if (debug) {
    std::cout << msg << std::endl;
  }
}

std::string util::ltrim(const std::string& s)
{
  size_t start = s.find_first_not_of(" \t\n\r\f\v");
  return (start == std::string::npos) ? "" : s.substr(start);
}

std::string util::rtrim(const std::string& s)
{
  size_t end = s.find_last_not_of(" \t\n\r\f\v");
  return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string util::trim(const std::string& s)
{
  return rtrim(ltrim(s));
}

bool BlueprintDecoder::checkLineIsEos(const std::string& line)
{
  std::string trimmedLine = util::trim(line);
  return trimmedLine == "}";
}

#pragma endregion