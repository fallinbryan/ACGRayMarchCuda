#pragma once

#include <glm/glm.hpp>

namespace raymarch {
  struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
  };


  enum class PrimitiveType {
    SPHERE,
    TRIANGLE,
    PLANE,
    BOX,
    DISC,
    ROUNDBOX,
    BOXFRAME,
    
  };

  enum class LightType {
    POINT,
    DIRECTIONAL,
    AREA_PLANE,
    AREA_DISC
  };

  enum class MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    EMISSIVE
  };

 struct Lambertian {
    Color albedo;
  };

struct Metal {
    Color albedo;
    float fuzz;
  };

struct Dielectric {
    float refractionIndex;
  };

struct Emissive {
    Color color;
    float intensity;
  };

  struct Material {
    MaterialType type;
    union {
      Lambertian lambertian;
      Metal metal;
      Dielectric dielectric;
      Emissive emissive;
    };
  };

  struct DirectionalLight {
    glm::vec3 direction;
    Color color;
  };

struct PointLight {
    glm::vec3 position;
    float radius;
  };

  struct AreaLightPlane {
    glm::vec3 origin;
    glm::vec3 normal;
    glm::vec3 u;
    glm::vec3 v;
  };

struct AreaLightDisc {
    glm::vec3 origin;
    glm::vec3 normal;
    float radius;
  };

  struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    float x;
    float y;
  };


  struct Sphere {
    glm::vec3 origin;
    float radius;
  };

  struct Box {
    glm::vec3 origin;
    glm::vec3 halfExtents;
  };

  struct RoundBox {
    glm::vec3 origin;
    glm::vec3 halfExtents;
    float edgeWidth;
  };

  struct BoxFrame {
    glm::vec3 origin;
    glm::vec3 halfExtents;
    float edgeWidth;
  };

  struct Triangle {
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
  };

  struct Disc {
    glm::vec3 origin;
    glm::vec3 normal;
    float radius;
  };

  struct Plane {
    glm::vec3 normal;
    glm::vec3 point;
  };

  struct Renderable {
    PrimitiveType type;
    Material material;
    union {
      Sphere sphere;
      Box box;
      RoundBox roundBox;
      BoxFrame boxFrame;
      Triangle triangle;
      Plane plane;
      Disc disc;
    };
    
    
  };

  struct Light {
    LightType type;
    glm::vec3 color;
    union {
      PointLight pointLight;
      DirectionalLight directionalLight;
      AreaLightPlane areaLightPlane;
      AreaLightDisc areaLightDisc;
    };
  };

} // namespace raymarch