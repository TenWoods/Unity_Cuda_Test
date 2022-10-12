#pragma once
#include "PluginAPI\IUnityGraphics.h"

#ifdef _WIN32
#include <Windows.h>
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include <fstream>

std::ofstream log_file;  //Debug: error output

class GraphicsResource
{
public: 
    int id;                               //texture id
    cudaGraphicsResource_t resource;      //cuda resource
    cudaArray_t array;
    size_t width;                         //texture width
    size_t height;                        //texture height
    size_t data_length;                   //texture data length
    void* data_pointer;                   //texture data pointer
    
    GraphicsResource(int id, size_t width, size_t height) : id(id), width(width), height(height)
    {
        resource = NULL;
        data_pointer = NULL;
        array = NULL;
        data_length = width * height * sizeof(uchar4);
    }

    void registerTexture();
    void mapCudaArray();
    void copyCudaArray();

    //data structure for compression
    //TODO
};
GraphicsResource* graphicsResource = NULL;

//Called by C#
extern "C"
{
    UNITY_INTERFACE_EXPORT void SendTextureIDToCuda(int texture_id, int width, int height);
    UNITY_INTERFACE_EXPORT void Dispose();
    UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc();
}

static void UNITY_INTERFACE_API OnRenderEvent(int eventID);

void CHECK_ERROR(cudaError_t err)
{
    if (!log_file.is_open())
        log_file.open("error_log.txt");
    if (err != cudaSuccess)
    {
        log_file << cudaGetErrorString(err) << std::endl;
    }
}

