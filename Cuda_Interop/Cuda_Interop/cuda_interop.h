#pragma once
#ifdef _WIN32
#include <Windows.h>
#endif

#include <fstream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "PluginAPI/IUnityGraphics.h"
#include "nvcomp/nvcomp.h"

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
    //data structure for compression
    void* temp_ptr;                       //temp pointer for nvcomp
    size_t temp_bytes;                    //temp data size
    void** device_compressed_ptrs;                       //out pointer for nvcomp
    size_t* device_compressed_bytes;                    //out data size
    size_t* uncompressed_bytes;
    void** uncompressed_ptrs;
    cudaStream_t  stream;                 //cuda stream

    
    GraphicsResource(int id, size_t width, size_t height) : id(id), width(width), height(height)
    {
        resource = NULL;
        data_pointer = NULL;
        array = NULL;
        isFirstDebug = true;
        isFirstCompress = true;
        data_length = width * height * sizeof(uchar4);
        cudaStreamCreate(&stream);
    }

    ~GraphicsResource()
    {
        cudaFree(data_pointer);
        cudaFree(temp_ptr);
        cudaFree(device_compressed_ptrs);
        cudaFree(device_compressed_bytes);
        cudaFree(uncompressed_bytes);
        cudaFree(uncompressed_ptrs);
    }

    void registerTexture();
    void mapResource();
    void copyCudaArray();
    void unmapResource();
    void unregisterResource();

    void compress();

private: 
    bool isFirstDebug;
    bool isFirstCompress;
    void output_for_debug();
    void output_decompress(size_t chunk_size, size_t batch_size);
    //void initialize_nvcomp();


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

std::string nvcompGetStatusString(nvcompStatus_t status)
{
    std::string status_string;
    switch (status)
    {
        case nvcompErrorInvalidValue:
            status_string = "invalid value";
            break;
        case nvcompErrorNotSupported:
            status_string = "not supported";
            break;
        case nvcompErrorCudaError:
            status_string = "cuda error";
            break;
        case nvcompErrorInternal:
            status_string = "internal";
            break;
        case nvcompErrorCannotDecompress:
            status_string = "can not decompress";
            break;
        default:
            status_string = "success";
            break;
    }
    return status_string;
}

void CHECK_ERROR(cudaError_t err, std::string filename, const int line)
{
    if (!log_file.is_open())
        log_file.open("error_log.txt");
    if (err != cudaSuccess)
    {
        log_file << "cuda error: " << '[' << cudaGetErrorString(err) << "] in " << filename << ", line" << line << std::endl;
    }
}

void CHECK_NVCOMP(nvcompStatus_t status, std::string filename, const int line)
{
    if (!log_file.is_open())
        log_file.open("error_log.txt");
    if (status != nvcompSuccess)
    {
        log_file << "nvcomp error: " << '[' << nvcompGetStatusString(status) << "] in " << filename << ", line" << line << std::endl;
    }
}

