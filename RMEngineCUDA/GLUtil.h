#pragma once
#include <GLFW/glfw3.h>
#include <string>

#include <exception>

#define GL_CHECK(x) x; checkGLError();

struct GLParams {
  int width;
  int height;
  GLuint vao;
  GLuint vbo;
  GLuint shader;
  GLuint texture;
  GLuint pixelBuffer;
  GLFWwindow* window;
};


std::string loadShaderSource(const std::string& filepath);

bool initGL(GLParams& params);

void updateTexture(GLParams& params);
void render(GLParams& params);

void checkGLError();

class OpenGLException : public std::exception {
private:
  std::string message;
public:
  OpenGLException(const std::string& message) : message(message) {}
  const char* what() const noexcept override {
    return message.c_str();
  }
};