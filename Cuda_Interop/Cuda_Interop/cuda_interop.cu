﻿#include "cuda_interop.h"
#include <iostream>
#include "nvcomp/cascaded.h"

void GraphicsResource::registerTexture()
{
	CHECK_ERROR(cudaGraphicsGLRegisterImage(&resource, id, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone), __FILE__, __LINE__);
}

void GraphicsResource::mapResource()
{
	CHECK_ERROR(cudaGraphicsMapResources(1, &resource, stream), __FILE__, __LINE__);
}

void GraphicsResource::copyCudaArray()
{
	CHECK_ERROR(cudaMalloc(&data_pointer, data_length), __FILE__, __LINE__);
	CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0), __FILE__, __LINE__);
    CHECK_ERROR(cudaMemcpy2DFromArray(data_pointer, width * sizeof(uchar4), array, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
    //Debug
	output_for_debug();
}

void GraphicsResource::unmapResource()
{
	CHECK_ERROR(cudaGraphicsUnmapResources(1, &resource, stream), __FILE__, __LINE__);
    //CHECK_ERROR(cudaStreamSynchronize(stream), __FILE__, __LINE__);
}

void GraphicsResource::unregisterResource()
{
	CHECK_ERROR(cudaGraphicsUnregisterResource(resource), __FILE__, __LINE__);
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

void GraphicsResource::compress()
{
    if (!isFirstCompress)
    {
        return;
    }
    isFirstCompress = false;
    if (!log_file.is_open())
        log_file.open("error_log.txt");
    size_t* host_uncompressed_bytes;
    const size_t chunk_size = 65536;
    const size_t batch_size = (data_length + chunk_size -1) / chunk_size;
    CHECK_ERROR(cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t) * batch_size), __FILE__, __LINE__);
    for (int i = 0; i < batch_size; i++)
    {
        if (i+1 < batch_size)
        {
            host_uncompressed_bytes[i] = chunk_size;
        }
        else
        {
            host_uncompressed_bytes[i] = data_length - (chunk_size*i);
        }
    }
    void** host_uncompressed_ptrs;
    CHECK_ERROR(cudaMallocHost(&host_uncompressed_ptrs, sizeof(size_t) * batch_size), __FILE__, __LINE__);
    for (int chunk_index = 0; chunk_index < batch_size; chunk_index++)
    {
        host_uncompressed_ptrs[chunk_index] = (char*)data_pointer + chunk_index * chunk_size;
    }

    cudaMalloc(&uncompressed_bytes, sizeof(size_t) * batch_size);
    cudaMalloc(&uncompressed_ptrs, sizeof(size_t) * batch_size);
    cudaMemcpyAsync(uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(uncompressed_ptrs, host_uncompressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

    CHECK_NVCOMP(nvcompBatchedCascadedCompressGetTempSize(batch_size, chunk_size, nvcompBatchedCascadedDefaultOpts, &temp_bytes), __FILE__, __LINE__);
    CHECK_ERROR(cudaMalloc(&temp_ptr, temp_bytes), __FILE__, __LINE__);
    size_t max_out_bytes;
    CHECK_NVCOMP(nvcompBatchedCascadedCompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedCascadedDefaultOpts, &max_out_bytes), __FILE__, __LINE__);
    void ** host_compressed_ptrs;
    cudaMallocHost(&host_compressed_ptrs, sizeof(size_t) * batch_size);
    for (int chunk_index = 0; chunk_index < batch_size; chunk_index++)
    {
        cudaMalloc(&host_compressed_ptrs[chunk_index], max_out_bytes);
    }
    cudaMalloc(&device_compressed_ptrs, sizeof(size_t) * batch_size);
    cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs,
            sizeof(size_t) * batch_size,cudaMemcpyHostToDevice, stream);
    cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size);

    CHECK_NVCOMP(nvcompBatchedCascadedCompressAsync(uncompressed_ptrs,
                                                    uncompressed_bytes,
                                                    chunk_size,
                                                    batch_size,
                                                    temp_ptr,
                                                    temp_bytes,
                                                    device_compressed_ptrs,
                                                    device_compressed_bytes,
                                                    nvcompBatchedCascadedDefaultOpts,
                                                    stream), __FILE__, __LINE__);
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
    nvcompBatchedCascadedGetDecompressSizeAsync(
            device_compressed_ptrs,
            device_compressed_bytes,
            uncompressed_bytes,
            batch_size,
            stream);
    nvcompStatus_t* device_statuses;
    cudaMalloc(&device_statuses, sizeof(nvcompStatus_t)*batch_size);
    size_t decomp_temp_bytes;
    CHECK_NVCOMP(nvcompBatchedCascadedDecompressGetTempSize(batch_size, chunk_size, &decomp_temp_bytes), __FILE__, __LINE__);
    void * device_decomp_temp;
    cudaMalloc(&device_decomp_temp, decomp_temp_bytes);
    size_t* device_actual_uncompressed_bytes;
    cudaMalloc(&device_actual_uncompressed_bytes, sizeof(size_t)*batch_size);
    CHECK_NVCOMP(nvcompBatchedCascadedDecompressAsync(device_compressed_ptrs,
                                                      device_compressed_bytes,
                                                      uncompressed_bytes,
                                                      device_actual_uncompressed_bytes,
                                                      batch_size,
                                                      device_decomp_temp,
                                                      decomp_temp_bytes,
                                                      uncompressed_ptrs,
                                                      device_statuses, stream), __FILE__, __LINE__);
    //debug start
    size_t* host_actual_uncompressed_bytes;
    cudaMallocHost(&host_actual_uncompressed_bytes, sizeof(size_t)*batch_size);
    cudaMemcpy(host_actual_uncompressed_bytes, device_actual_uncompressed_bytes, sizeof(size_t)*batch_size, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < batch_size; i++)
//    {
//        log_file << "decompressed bytes: " << *(host_actual_uncompressed_bytes + i) << std::endl;
//    }
    //debug end
    output_decompress(batch_size, host_actual_uncompressed_bytes);
}

void GraphicsResource::output_decompress(size_t batch_size, const size_t* host_uncompressed_bytes)
{
    void** test;
    cudaMallocHost(&test, sizeof(size_t) * batch_size);
    void** host_uncompressed_ptrs;
    cudaMallocHost(&host_uncompressed_ptrs, sizeof(size_t) * batch_size);
    cudaMemcpy(host_uncompressed_ptrs, uncompressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch_size; i++)
    {
        //log_file << host_uncompressed_bytes[i] << std::endl;
        CHECK_ERROR(cudaMallocHost(test+i, host_uncompressed_bytes[i]), __FILE__, __LINE__);
        CHECK_ERROR(cudaMemcpyAsync(test[i], host_uncompressed_ptrs[i] , host_uncompressed_bytes[i], cudaMemcpyDeviceToHost, stream), __FILE__, __LINE__);
    }
    cudaStreamSynchronize(stream);
    std::ofstream file;
    file.open("debug_decompress.ppm");
    file << "P3" << std::endl
        << "1920 1080" << std::endl
        << "255" << std::endl;
    for (int i = 0; i < batch_size; i++)
    {
        int batch_image_size = host_uncompressed_bytes[i]/sizeof(unsigned char)/4;
        for (int j = 0; j < batch_image_size; j++)
        {
            unsigned char* c = (unsigned char*)test[i] + j * 4;
            file << (int)*c << ' '
                 << (int)*(c + 1) << ' '
                 << (int)*(c + 2) << std::endl;
        }
    }
    //cudaFree(test);
//    file.close();
}

UNITY_INTERFACE_EXPORT void SendTextureIDToCuda(int texture_id, int width, int height)
{
    if (graphicsResource == NULL)
    {
        graphicsResource = new GraphicsResource(texture_id, width, height);
    }
}

static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
	graphicsResource->registerTexture();
	graphicsResource->mapResource();
	graphicsResource->copyCudaArray();
    graphicsResource->compress();
	graphicsResource->unmapResource();
	graphicsResource->unregisterResource();
}

UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc()
{
	return OnRenderEvent;
}

UNITY_INTERFACE_EXPORT void Dispose()
{
	cudaGraphicsUnmapResources(1, &graphicsResource->resource, 0);
    //cudaStreamSynchronize(graphicsResource->stream);
	log_file.close();
}

