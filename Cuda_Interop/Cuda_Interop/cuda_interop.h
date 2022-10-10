#pragma once
#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include <map>
#define DllExport __declspec(dllexport)

struct GraphicsResource
{
    int id;                               //texture id
    cudaGraphicsResource_t resource;      //cuda resource
    size_t width;                         //texture width
    size_t height;                        //texture height
    size_t data_length;                   //texture data length
    void* data_pointer;                   //texture data pointer

    //data structure for compression
    //TODO
};

std::map<int, GraphicsResource*> graphicsResources;

//called by C#
extern "C"
{
	DllExport int Test(int input);
    DllExport void SendTextureToCuda(int texture_id, int width, int height);
}

void readTexture(int texture_id, int width, int height);
GraphicsResource* getResource(int texture_id);
