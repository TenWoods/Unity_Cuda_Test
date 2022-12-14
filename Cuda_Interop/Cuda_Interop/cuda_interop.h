#pragma once
#ifdef _WIN32
#include <Windows.h>
#endif

#include <fstream>
#include <string>
#include <map>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "PluginAPI/IUnityGraphics.h"
#include "nvcomp/nvcomp.h"

std::ofstream log_file;  //Debug: error output
std::ofstream fifo_file;
std::ofstream  output_file;

class GraphicsResource
{
public:
    //data structure for compression
//    void* temp_ptr;                       //temp pointer for nvcomp
//    size_t temp_bytes;                    //temp data size
//    void** device_compressed_ptrs;                       //out pointer for nvcomp
//    size_t* device_compressed_bytes;                    //out data size
//    size_t* uncompressed_bytes;
//    void** uncompressed_ptrs;
    // src device memory
    void *device_src = nullptr;
    size_t src_capacity = 0;
    size_t src_limit = 0;

    // workspace device memory
    void *device_workspace = nullptr;
    size_t workspace_capacity = 0;

    // dst device memory
    void *device_dst = nullptr;
    size_t dst_capacity = 0;
    size_t dst_limit = 0;

    cudaStream_t  stream;                 //cuda stream
    bool isMapped;
    bool isRegistered;
    int count = 0;
    
    GraphicsResource(int id, int type, size_t width, size_t height, int processID, int cameraID) :
    id(id), type((texture_type)type), width(width), height(height), processID(processID), cameraID(cameraID)
    {
        resource = nullptr;
        data_pointer = nullptr;
        array = nullptr;
        data_length = width * height * sizeof(uchar4);
        //cudaStreamCreate(&stream);
        isMapped = false;
        isRegistered = false;
    }

    ~GraphicsResource()
    {
        cudaFree(data_pointer);
        cudaFree(device_src);
        cudaFree(device_workspace);
        cudaFree(device_dst);
        unmapResource();
        unregisterResource();
//        cudaFree(temp_ptr);
//        cudaFree(device_compressed_ptrs);
//        cudaFree(device_compressed_bytes);
//         cudaFree(uncompressed_bytes);
//        cudaFree(uncompressed_ptrs);
    }

    void registerTexture();
    void mapResource();
    void copyCudaArray();
    void unmapResource();
    void unregisterResource();
    void resizeDeviceMemory(size_t src_capacity, size_t workspace_capacity, size_t dst_capacity);
    void compress();
    void sendData();
    //void decompress();

private:
    int id;                               //texture id
    cudaGraphicsResource_t resource;      //cuda resource
    cudaArray_t array;
    enum texture_type
    {
        Color = 0,
        Depth = 1
    } type;
    size_t width;                         //texture width
    size_t height;                        //texture height
    int processID;                        //process belong to
    int cameraID;                         //camera belong to
    size_t data_length;                   //texture data length
    void* data_pointer;                   //texture data pointer
//    bool isFirstDebug;
//    bool isFirstCompress;
    //void output_for_debug();
    //void output_decompress(/*size_t batch_size, const size_t* host_uncompressed_bytes*/);
    //void initialize_nvcomp();


};

std::map<int, GraphicsResource*> resources;

//Called by C#
extern "C"
{
    UNITY_INTERFACE_EXPORT void SendTextureIDToCuda(int texture_id, int type, int width, int height, int processID, int cameraID);
    UNITY_INTERFACE_EXPORT void Dispose();
    UNITY_INTERFACE_EXPORT void GenerateNamedPipe(int processID, int cameraID);
    UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetPostRenderFunc();
    static void UNITY_INTERFACE_API OnRenderEvent(int eventID);
}

std::string nvcompGetStatusString(nvcompError_t status)
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

void CHECK_NVCOMP(nvcompError_t status, std::string filename, const int line)
{
    if (!log_file.is_open())
        log_file.open("error_log.txt");
    if (status != nvcompSuccess)
    {
        log_file << "nvcomp error: " << '[' << nvcompGetStatusString(status) << "] in " << filename << ", line" << line << std::endl;
    }
}

