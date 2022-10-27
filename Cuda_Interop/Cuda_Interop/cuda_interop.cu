#include "cuda_interop.h"
#include "cuda_runtime_api.h"
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
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
//        count++;
//        if (!log_file.is_open())
//            log_file.open("debug_log.txt");
//        log_file << count << std::endl;
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

void GraphicsResource::output_for_debug()
{
	if (!isFirstDebug)
		return;
	isFirstDebug = false;
	std::ofstream file;
	file.open("debug.ppm");
	void* test = malloc(data_length);
	CHECK_ERROR(cudaMemcpy(test, data_pointer, data_length, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	file << "P3" << std::endl
		<< "1920 1080" << std::endl
		<< "255" << std::endl;
	int texture_size = width * height;
	for (int i = 0; i < texture_size; i++)
	{
		unsigned char* c = (unsigned char*)test + i * 4;
		file << (int)*c << ' '
			<< (int)*(c + 1) << ' '
			<< (int)*(c + 2) << std::endl;
	}
	free(test);
	file.close();
}

void GraphicsResource::resizeDeviceMemory(size_t src_capacity, size_t workspace_capacity, size_t dst_capacity)
{
    if (this->src_capacity < src_capacity)
    {
        if (this->device_src != NULL)
        {
            CHECK_ERROR(cudaFree(this->device_src), __FILE__, __LINE__);
        }
        CHECK_ERROR(cudaMalloc(&this->device_src, src_capacity + 10), __FILE__, __LINE__);
        this->src_capacity = src_capacity;
    }
    if (this->workspace_capacity < workspace_capacity)
    {
        if (this->device_workspace != NULL)
        {
            CHECK_ERROR(cudaFree(this->device_workspace), __FILE__, __LINE__);
        }
        CHECK_ERROR(cudaMalloc(&this->device_workspace, workspace_capacity), __FILE__, __LINE__);
        this->workspace_capacity = workspace_capacity;
    }
    if (this->dst_capacity < dst_capacity)
    {
        if (this->device_dst != NULL)
        {
            CHECK_ERROR(cudaFree(this->device_dst), __FILE__, __LINE__);
        }
        CHECK_ERROR(cudaMalloc(&this->device_dst, dst_capacity), __FILE__, __LINE__);
        this->dst_capacity = dst_capacity;
    }
}

//old version
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
//    nvcompCascadedFormatOpts format;
//    format.num_RLEs = 1;
//    format.num_deltas = 1;
//    format.use_bp = 1;
    nvcompType_t type = NVCOMP_TYPE_UINT;

//    void* metadata_ptr;
    size_t workspace_capacity;  //temp_bytes
    size_t dst_capacity;        //out_bytes =
    size_t metadata_bytes;      //in_bytes =

    CHECK_NVCOMP(nvcompCascadedCompressConfigure(
            NULL,
            type,
            src_limit,
            &metadata_bytes,
            &workspace_capacity,
            &dst_capacity
    ), __FILE__, __LINE__);
    resizeDeviceMemory(0, workspace_capacity, dst_capacity);
    CHECK_NVCOMP(nvcompCascadedCompressAsync(
            NULL,
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
    void* host_result;
    cudaMallocHost(&host_result, dst_limit);
    cudaMemcpy(host_result, device_dst, dst_limit, cudaMemcpyDeviceToHost);
    fifo_file.open("output_to_java", std::ios::out | std::ios::binary);
    fifo_file.write((char*)host_result, dst_limit/sizeof(char));
    fifo_file.flush();
    fifo_file.close();
    unmapResource();
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

//new version
//void GraphicsResource::compress()
//{
//    if (!isMapped)
//        return;
//    if (!isFirstCompress)
//    {
//        return;
//    }
//    isFirstCompress = false;
//    size_t* host_uncompressed_bytes;
//    const size_t chunk_size = 4096;
//    const size_t batch_size = (data_length + chunk_size -1) / chunk_size;
//    CHECK_ERROR(cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t) * batch_size), __FILE__, __LINE__);
//    for (int i = 0; i < batch_size; i++)
//    {
//        if (i+1 < batch_size)
//        {
//            host_uncompressed_bytes[i] = chunk_size;
//        }
//        else
//        {
//            host_uncompressed_bytes[i] = data_length - (chunk_size*i);
//        }
//    }
//    void** host_uncompressed_ptrs;
//    CHECK_ERROR(cudaMallocHost(&host_uncompressed_ptrs, sizeof(size_t) * batch_size), __FILE__, __LINE__);
//    for (int chunk_index = 0; chunk_index < batch_size; chunk_index++)
//    {
//        host_uncompressed_ptrs[chunk_index] = (char*)data_pointer + chunk_index * chunk_size;
//    }
//    if (uncompressed_ptrs == nullptr)
//    {
//        cudaMalloc(&uncompressed_bytes, sizeof(size_t) * batch_size);
//    }
//    if (uncompressed_bytes == nullptr)
//    {
//        cudaMalloc(&uncompressed_ptrs, sizeof(size_t) * batch_size);
//    }
//    cudaMemcpyAsync(uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
//    cudaMemcpyAsync(uncompressed_ptrs, host_uncompressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
//    //free ptr
//    cudaFree(host_uncompressed_bytes);
//    cudaFree(host_uncompressed_ptrs);
//
//    CHECK_NVCOMP(nvcompBatchedCascadedCompressGetTempSize(batch_size, chunk_size, nvcompBatchedCascadedDefaultOpts, &temp_bytes), __FILE__, __LINE__);
//    CHECK_ERROR(cudaMalloc(&temp_ptr, temp_bytes), __FILE__, __LINE__);
//    size_t max_out_bytes;
//    CHECK_NVCOMP(nvcompBatchedCascadedCompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedCascadedDefaultOpts, &max_out_bytes), __FILE__, __LINE__);
//    void ** host_compressed_ptrs;
//    cudaMallocHost(&host_compressed_ptrs, sizeof(size_t) * batch_size);
//    for (int chunk_index = 0; chunk_index < batch_size; chunk_index++)
//    {
//        cudaMalloc(&host_compressed_ptrs[chunk_index], max_out_bytes);
//    }
//    if (device_compressed_ptrs == nullptr)
//    {
//        cudaMalloc(&device_compressed_ptrs, sizeof(size_t) * batch_size);
//    }
//    cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs,
//            sizeof(size_t) * batch_size,cudaMemcpyHostToDevice, stream);
//
//    //CHECK_ERROR(cudaStreamSynchronize(stream), __FILE__, __LINE__);
//    if (device_compressed_bytes == nullptr)
//    {
//        cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size);
//    }
//    CHECK_NVCOMP(nvcompBatchedCascadedCompressAsync(uncompressed_ptrs,
//                                                    uncompressed_bytes,
//                                                    chunk_size,
//                                                    batch_size,
//                                                    temp_ptr,
//                                                    temp_bytes,
//                                                    device_compressed_ptrs,
//                                                    device_compressed_bytes,
//                                                    nvcompBatchedCascadedDefaultOpts,
//                                                    stream), __FILE__, __LINE__);
//    cudaStreamSynchronize(stream);
//    cudaFree(temp_ptr);
//    cudaFree(data_pointer);
//    cudaFree(uncompressed_ptrs);
//    cudaFree(uncompressed_bytes);


    //debug start
//    size_t* host_out_bytes;
//    cudaMallocHost(&host_out_bytes, sizeof(size_t)*batch_size);
//    cudaMemcpy(host_out_bytes, device_compressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < batch_size; i++)
//    {
//        log_file << "compressed bytes: " << *(host_out_bytes + i) << std::endl;
//    }
    //debug end

    //decompress
//    nvcompBatchedCascadedGetDecompressSizeAsync(
//            device_compressed_ptrs,
//            device_compressed_bytes,
//            uncompressed_bytes,
//            batch_size,
//            stream);
//    nvcompStatus_t* device_statuses;
//    cudaMalloc(&device_statuses, sizeof(nvcompStatus_t)*batch_size);
//    size_t decomp_temp_bytes;
//    CHECK_NVCOMP(nvcompBatchedCascadedDecompressGetTempSize(batch_size, chunk_size, &decomp_temp_bytes), __FILE__, __LINE__);
//    void * device_decomp_temp;
//    cudaMalloc(&device_decomp_temp, decomp_temp_bytes);
//    size_t* device_actual_uncompressed_bytes;
//    cudaMalloc(&device_actual_uncompressed_bytes, sizeof(size_t)*batch_size);
//    CHECK_NVCOMP(nvcompBatchedCascadedDecompressAsync(device_compressed_ptrs,
//                                                      device_compressed_bytes,
//                                                      uncompressed_bytes,
//                                                      device_actual_uncompressed_bytes,
//                                                      batch_size,
//                                                      device_decomp_temp,
//                                                      decomp_temp_bytes,
//                                                      uncompressed_ptrs,
//                                                      device_statuses, stream), __FILE__, __LINE__);
    //debug start
//    size_t* host_actual_uncompressed_bytes;
//    cudaMallocHost(&host_actual_uncompressed_bytes, sizeof(size_t)*batch_size);
//    cudaMemcpy(host_actual_uncompressed_bytes, device_actual_uncompressed_bytes, sizeof(size_t)*batch_size, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < batch_size; i++)
//    {
//        log_file << "decompressed bytes: " << *(host_actual_uncompressed_bytes + i) << std::endl;
//    }
    //debug end
    //output_decompress(batch_size, host_actual_uncompressed_bytes);
//}

void GraphicsResource::output_decompress(/*size_t batch_size, const size_t* host_uncompressed_bytes*/)
{
    std::ofstream file;
    file.open("debug.ppm");
    void* test = malloc(src_limit);
    CHECK_ERROR(cudaMemcpy(test, device_src, src_limit, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    file << "P3" << std::endl
         << "1920 1080" << std::endl
         << "255" << std::endl;
    int texture_size = width * height;
    for (int i = 0; i < texture_size; i++)
    {
        unsigned char* c = (unsigned char*)test + i * 4;
        file << (int)*c << ' '
             << (int)*(c + 1) << ' '
             << (int)*(c + 2) << std::endl;
    }
    free(test);
    file.close();
//    void** test;
//    cudaMallocHost(&test, sizeof(size_t) * batch_size);
//    void** host_uncompressed_ptrs;
//    cudaMallocHost(&host_uncompressed_ptrs, sizeof(size_t) * batch_size);
//    cudaMemcpy(host_uncompressed_ptrs, uncompressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < batch_size; i++)
//    {
//        //log_file << host_uncompressed_bytes[i] << std::endl;
//        CHECK_ERROR(cudaMallocHost(test+i, host_uncompressed_bytes[i]), __FILE__, __LINE__);
//        CHECK_ERROR(cudaMemcpyAsync(test[i], host_uncompressed_ptrs[i] , host_uncompressed_bytes[i], cudaMemcpyDeviceToHost, stream), __FILE__, __LINE__);
//    }
//    cudaStreamSynchronize(stream);
//    std::ofstream file;
//    file.open("debug_decompress.ppm");
//    file << "P3" << std::endl
//        << "1920 1080" << std::endl
//        << "255" << std::endl;
//    for (int i = 0; i < batch_size; i++)
//    {
//        int batch_image_size = host_uncompressed_bytes[i]/sizeof(unsigned char)/4;
//        for (int j = 0; j < batch_image_size; j++)
//        {
//            unsigned char* c = (unsigned char*)test[i] + j * 4;
//            file << (int)*c << ' '
//                 << (int)*(c + 1) << ' '
//                 << (int)*(c + 2) << std::endl;
//        }
//    }
    //cudaFree(test);
//    file.close();
}

UNITY_INTERFACE_EXPORT void SendTextureIDToCuda(int texture_id, int width, int height)
{
    if (graphicsResource == nullptr)
    {
        graphicsResource = new GraphicsResource(texture_id, width, height);
        //GenerateNamedPipe();
    }
}

UNITY_INTERFACE_EXPORT void GenerateNamedPipe()
{
    const char* fifo_name = "output_to_java";
    int res = mkfifo(fifo_name, 0777);
    if (res != 0 && errno != 17)
        return;
    fifo_file.open(fifo_name, std::ios::out | std::ios::binary);
    fifo_file << "test output";
    fifo_file.flush();
    fifo_file.close();
}

static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
	//graphicsResource->registerTexture();
	//graphicsResource->mapResource();
	//graphicsResource->copyCudaArray();
    graphicsResource->compress();
    //graphicsResource->decompress();
	//graphicsResource->unmapResource();
}

UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc()
{
	return OnRenderEvent;
}

UNITY_INTERFACE_EXPORT void Dispose()
{
    delete graphicsResource;
	log_file.close();
    fifo_file.close();
}

