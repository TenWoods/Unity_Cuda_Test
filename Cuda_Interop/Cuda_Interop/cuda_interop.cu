#include "cuda_interop.h"
#include "cuda_runtime_api.h"
#include <iostream>
#include <sys/stat.h>
#include <string>
#include "nvcomp/cascaded.h"

void GraphicsResource::registerTexture()
{
    if (!isRegistered)
    {
        CHECK_ERROR(cudaGraphicsGLRegisterImage(&resource, id, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone), __FILE__, __LINE__);
        cudaStreamCreate(&stream);
        cudaDeviceSynchronize();
        isRegistered = true;
    }
}

void GraphicsResource::mapResource()
{
    if (!isMapped)
    {
        CHECK_ERROR(cudaGraphicsMapResources(1, &resource, stream), __FILE__, __LINE__);
        CHECK_ERROR(cudaStreamSynchronize(stream), __FILE__, __LINE__);
        isMapped = true;
    }
}

void GraphicsResource::copyCudaArray()
{
    if (data_pointer == nullptr)
    {
        CHECK_ERROR(cudaMalloc(&data_pointer, data_length), __FILE__, __LINE__);
    }
	CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0), __FILE__, __LINE__);
    CHECK_ERROR(cudaMemcpy2DFromArray(data_pointer, width * sizeof(uchar4), array, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
    //Debug
	//output_for_debug();
}

void GraphicsResource::unmapResource()
{
    if (!isRegistered || !isMapped)
        return;
	CHECK_ERROR(cudaGraphicsUnmapResources(1, &resource, stream), __FILE__, __LINE__);
    CHECK_ERROR(cudaStreamSynchronize(stream), __FILE__, __LINE__);
    isMapped = false;
}

void GraphicsResource::unregisterResource()
{
    if (!isRegistered)
        return;
	CHECK_ERROR(cudaGraphicsUnregisterResource(resource), __FILE__, __LINE__);
    isRegistered = false;
}

//void GraphicsResource::output_for_debug()
//{
//	if (!isFirstDebug)
//		return;
//	isFirstDebug = false;
//	std::ofstream file;
//	file.open("debug.ppm");
//	void* test = malloc(data_length);
//	CHECK_ERROR(cudaMemcpy(test, data_pointer, data_length, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
//	file << "P3" << std::endl
//		<< "1920 1080" << std::endl
//		<< "255" << std::endl;
//	int texture_size = width * height;
//	for (int i = 0; i < texture_size; i++)
//	{
//		unsigned char* c = (unsigned char*)test + i * 4;
//		file << (int)*c << ' '
//			<< (int)*(c + 1) << ' '
//			<< (int)*(c + 2) << std::endl;
//	}
//	free(test);
//	file.close();
//}

void GraphicsResource::resizeDeviceMemory(size_t src_capacity, size_t workspace_capacity, size_t dst_capacity)
{
    if (this->src_capacity < src_capacity)
    {
        if (this->device_src != nullptr)
        {
            CHECK_ERROR(cudaFree(this->device_src), __FILE__, __LINE__);
        }
        CHECK_ERROR(cudaMalloc(&this->device_src, src_capacity + 10), __FILE__, __LINE__);
        this->src_capacity = src_capacity;
    }
    if (this->workspace_capacity < workspace_capacity)
    {
        if (this->device_workspace != nullptr)
        {
            CHECK_ERROR(cudaFree(this->device_workspace), __FILE__, __LINE__);
        }
        CHECK_ERROR(cudaMalloc(&this->device_workspace, workspace_capacity), __FILE__, __LINE__);
        this->workspace_capacity = workspace_capacity;
    }
    if (this->dst_capacity < dst_capacity)
    {
        if (this->device_dst != nullptr)
        {
            CHECK_ERROR(cudaFree(this->device_dst), __FILE__, __LINE__);
        }
        CHECK_ERROR(cudaMalloc(&this->device_dst, dst_capacity), __FILE__, __LINE__);
        this->dst_capacity = dst_capacity;
    }
}

void GraphicsResource::compress()
{
    if (!isRegistered)
    {
        registerTexture();
    }
    cudaDeviceSynchronize();
    if (!isMapped)
    {
        mapResource();
    }
    copyCudaArray();
    if (!log_file.is_open())
        log_file.open("error_log.txt");
    log_file << '[' << count++ << ']' << ' ';
    std::cout << '[' << count << ']' << std::endl;
    src_limit = data_length;
    nvcompType_t type = NVCOMP_TYPE_UINT;

//    void* metadata_ptr;
    size_t workspace_capacity;  //temp_bytes
    size_t dst_capacity;        //out_bytes
    size_t metadata_bytes;      //in_bytes

    CHECK_NVCOMP(nvcompCascadedCompressConfigure(
            nullptr,
            type,
            src_limit,
            &metadata_bytes,
            &workspace_capacity,
            &dst_capacity
    ), __FILE__, __LINE__);
    resizeDeviceMemory(0, workspace_capacity, dst_capacity);
    CHECK_NVCOMP(nvcompCascadedCompressAsync(
            nullptr,
            type,
            data_pointer,       //uncompressed_ptr
            src_limit,             //uncompressed_bytes
            device_workspace,      //temp_ptr
            this->workspace_capacity,    //temp_bytes
            device_dst,            //compressed_ptr
            &dst_limit,            //compressed_bytes
            stream
    ), __FILE__, __LINE__);
    log_file << src_limit << "->" << dst_limit << std::endl;
//    if (isFirstDebug)
//    {
//        isFirstDebug = false;
//        output_file.open("output.txt", std::ios::in | std::ios::binary);
//        output_file.write((char*)host_result, dst_limit);
//        output_file.close();
//    }
    //cudaFree(host_result);

    //decompress for debug
    //resizeDeviceMemory(dst_limit, 0, 0);
//    if (isFirstCompress) {
//        CHECK_NVCOMP(nvcompCascadedDecompressConfigure(
//                device_dst,
//                dst_limit,
//                &metadata_ptr,
//                &metadata_bytes,
//                &workspace_capacity,
//                &dst_capacity,
//                stream
//        ), __FILE__, __LINE__);
//        resizeDeviceMemory(dst_capacity, workspace_capacity, 0);
//        CHECK_NVCOMP(nvcompCascadedDecompressAsync(
//                device_dst,
//                dst_limit,
//                metadata_ptr,
//                metadata_bytes,
//                device_workspace,
//                this->workspace_capacity,
//                device_src,
//                src_capacity,
//                stream
//        ), __FILE__, __LINE__);
//        output_decompress();
//        isFirstCompress = false;
//    }
}

void GraphicsResource::sendData()
{
    void* host_result;
    cudaMallocHost(&host_result, dst_limit);
    cudaMemcpy(host_result, device_dst, dst_limit, cudaMemcpyDeviceToHost);
    std::string fifo_name;
    switch (type)
    {
        case Color:
            fifo_name = "output_to_java_color_" + std::to_string(processID) + std::to_string(cameraID);
            break;
        case Depth:
            fifo_name = "output_to_java_depth_" + std::to_string(processID) + std::to_string(cameraID);
            break;
        default:
            break;
    }
    fifo_file.open(fifo_name, std::ios::out | std::ios::binary);
    fifo_file.write((char*)host_result, dst_limit/sizeof(char));
    fifo_file.flush();
    fifo_file.close();
    unmapResource();
}

//void GraphicsResource::output_decompress(/*size_t batch_size, const size_t* host_uncompressed_bytes*/)
//{
//    std::ofstream file;
//    file.open("debug.ppm");
//    void* test = malloc(src_limit);
//    CHECK_ERROR(cudaMemcpy(test, device_src, src_limit, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
//    file << "P3" << std::endl
//         << "1920 1080" << std::endl
//         << "255" << std::endl;
//    int texture_size = width * height;
//    for (int i = 0; i < texture_size; i++)
//    {
//        unsigned char* c = (unsigned char*)test + i * 4;
//        file << (int)*c << ' '
//             << (int)*(c + 1) << ' '
//             << (int)*(c + 2) << std::endl;
//    }
//    free(test);
//    file.close();
//}

UNITY_INTERFACE_EXPORT void SendTextureIDToCuda(int texture_id, int type, int width, int height, int processID, int cameraID)
{
    if (resources.find(texture_id) != resources.end())
    {
        return;
    }
    GraphicsResource* resource = new GraphicsResource(texture_id, type, width, height, processID, cameraID);
    resources[texture_id] = resource;
}

UNITY_INTERFACE_EXPORT void GenerateNamedPipe(int processID, int cameraID)
{
    if (processID == -1)
    {
        if (!log_file.is_open())
        {
            log_file.open("error_log");
        }
        log_file << "illegal processID" << std::endl;
        return;
    }
    std::string fifo_name = "output_to_java_color_" + std::to_string(processID) + std::to_string(cameraID);
    //create color fifo file
    int res = mkfifo(fifo_name.c_str(), 0777);
    if (res != 0 && errno != 17)
    {
        if (!log_file.is_open())
        {
            log_file.open("error_log");
        }
        log_file << "color fifo error!" << std::endl;
        return;
    }
    //create depth fifo file
    fifo_name = "output_to_java_depth_" + std::to_string(processID) + std::to_string(cameraID);
    res = mkfifo(fifo_name.c_str(), 0777);
    if (res != 0 && errno != 17)
    {
        if (!log_file.is_open())
        {
            log_file.open("error_log");
        }
        log_file << "depth fifo error!" << std::endl;
        return;
    }
}

UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetPostRenderFunc()
{
    return OnRenderEvent;
}

static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
    resources[eventID]->compress();
    resources[eventID]->sendData();
}

UNITY_INTERFACE_EXPORT void Dispose()
{
    std::map<int, GraphicsResource*>::iterator i;
    for(i = resources.begin(); i != resources.end(); i++)
    {
        delete i->second;
    }
	log_file.close();
    fifo_file.close();
}

