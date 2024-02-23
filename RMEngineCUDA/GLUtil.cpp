#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "GLUtil.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>




std::string loadShaderSource(const std::string& filepath) {
  std::ifstream shaderFile(filepath);
  if (!shaderFile) {
    std::cerr << "Failed to open shader file: " << filepath << std::endl;
    return "";
  }
  std::string source((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
  return source;
}

GLuint compileShader(unsigned int type, const std::string& source) {
  GLuint id = glCreateShader(type);
  const char* src = source.c_str();
  glShaderSource(id, 1, &src, nullptr);
  glCompileShader(id);

  int result;
  glGetShaderiv(id, GL_COMPILE_STATUS, &result);
  if (result == GL_FALSE) {
    int length;
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
    char* message = (char*)alloca(length * sizeof(char));
    glGetShaderInfoLog(id, length, &length, message);
    std::cerr << "Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader!" << std::endl;
    std::cerr << message << std::endl;
    glDeleteShader(id);
    throw OpenGLException("Failed to compile shader");
    return 0;
  }

  return id;
}

GLuint createShader(const std::string& vertexShader, const std::string& fragmentShader) {
  GLuint program = glCreateProgram();
  GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShader);
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShader);

  GL_CHECK(glAttachShader(program, vs));
  GL_CHECK(glAttachShader(program, fs));
  GL_CHECK(glLinkProgram(program));
  GL_CHECK(glValidateProgram(program));
  GL_CHECK(glDeleteShader(vs));
  GL_CHECK(glDeleteShader(fs));

  return program;
}

bool initGL(GLParams &params) {
 
  float vertices[] = {
    // Positions     // Texture Coords (flipped Y axis)
    -1.0f,  1.0f,  0.0f, 0.0f, // Top Left
    -1.0f, -1.0f,  0.0f, 1.0f, // Bottom Left
     1.0f, -1.0f,  1.0f, 1.0f, // Bottom Right

    -1.0f,  1.0f,  0.0f, 0.0f, // Top Left
     1.0f, -1.0f,  1.0f, 1.0f, // Bottom Right
     1.0f,  1.0f,  1.0f, 0.0f  // Top Right
  };
  std::string vertShader = loadShaderSource("basicVertShader.glsl");
  std::string fragShader = loadShaderSource("basicFragShader.glsl");

  // Initialize GLFW
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow* window = glfwCreateWindow(params.width, params.height, "Ray Marching V2", NULL, NULL);
  if (window == NULL)
  {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  // Initialize GLAD
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return false;
  }

  glViewport(0, 0, params.width, params.height);

  GLuint textureId;
  GL_CHECK(glGenTextures(1, &textureId));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureId));
  GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, params.width, params.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
  //GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));

  GLuint pixelBuffer;
  GL_CHECK(glGenBuffers(1, &pixelBuffer));
  GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer));
  GL_CHECK(glBufferData(GL_PIXEL_UNPACK_BUFFER, params.width * params.height * 4, NULL, GL_STREAM_DRAW));
  GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

  GLuint shader = createShader(vertShader, fragShader);

  GLuint quadVAO, quadVBO;
  GL_CHECK(glGenVertexArrays(1, &quadVAO));
  GL_CHECK(glGenBuffers(1, &quadVBO));
  GL_CHECK(glBindVertexArray(quadVAO));
  GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, quadVBO));
  GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW));
  GL_CHECK(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0));
  GL_CHECK(glEnableVertexAttribArray(0));
  GL_CHECK(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float))));
  GL_CHECK(glEnableVertexAttribArray(1));

  params.shader = shader;
  params.vao = quadVAO;
  params.texture = textureId;
  params.vbo = quadVBO;
  params.pixelBuffer = pixelBuffer;
  params.window = window;

  return true;
}

void render(GLParams &params) {
  
  GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));

  GL_CHECK(glUseProgram(params.shader));
  GL_CHECK(glBindVertexArray(params.vao));
  GL_CHECK(glActiveTexture(GL_TEXTURE0));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, params.texture));
  GLint textureLocation = glGetUniformLocation(params.shader, "screenTexture");
  GL_CHECK(glUniform1i(textureLocation, 0));
  GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 6));
  GL_CHECK(glfwSwapBuffers(params.window));
  GL_CHECK(glfwPollEvents());
}

void updateTexture(GLParams &params) {
  GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, params.pixelBuffer));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, params.texture));
  GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, params.width, params.height, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
  GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
};



void checkGLError() {
  GLenum error = glGetError();
if (error != GL_NO_ERROR) {
  std::string errorString;
    switch(error) {
      case GL_INVALID_ENUM:
        errorString = "GL_INVALID_ENUM";
        break;
      case GL_INVALID_VALUE:
        errorString = "GL_INVALID_VALUE";
        break;
      case GL_INVALID_OPERATION:
        errorString = "GL_INVALID_OPERATION";
        break;
      case GL_STACK_OVERFLOW:
        errorString = "GL_STACK_OVERFLOW";
        break;
      case GL_STACK_UNDERFLOW:
        errorString = "GL_STACK_UNDERFLOW";
        break;
      case GL_OUT_OF_MEMORY:
        errorString = "GL_OUT_OF_MEMORY";
        break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:
        errorString = "GL_INVALID_FRAMEBUFFER_OPERATION";
        break;
      default:
        errorString = "UNKNOWN";
        break;
    }
    throw OpenGLException(errorString);
   
  }
}