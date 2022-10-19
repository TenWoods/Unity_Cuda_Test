#include "cuda_interop.h"
#include <iostream>
#include "nvcomp/cascaded.h"

void GraphicsResource::registerTexture()
{
	CHECK_ERROR(cudaGraphicsGLRegisterImage(&resource, id, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone), __FILE__, __LINE__);
}

void GraphicsResource::mapResource()
{
	CHECK_ERROR(cudaGraphicsMapResources(1, &resource, 0), __FILE__, __LINE__);
}

void GraphicsResource::copyCudaArray()
{
	CHECK_ERROR(cudaMalloc(&data_pointer, data_length), __FILE__, __LINE__);
	CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0), __FILE__, __LINE__);
	CHECK_ERROR(cudaMemcpy2DFromArray(data_pointer, width * sizeof(uchar4), array, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
	//Debug
	//output_for_debug();
}

void GraphicsResource::unmapResource()
{
	CHECK_ERROR(cudaGraphicsUnmapResources(1, &resource, 0), __FILE__, __LINE__);
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
    size_t* uncompressed_batch;
    const size_t chunk_size = 65536;
    const size_t batch_size = (data_length + chunk_size -1) / chunk_size;
    cudaMalloc(&uncompressed_batch, sizeof(size_t) * batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        if (i+1 < batch_size)
        {
            uncompressed_batch[i] = chunk_size;
        }
        else
        {
            uncompressed_batch[i] = data_length - (chunk_size*i);
        }
    }
    void** uncompressed_ptrs;
    cudaMalloc(&uncompressed_ptrs, sizeof(size_t) * batch_size);
    for (int chunk_index = 0; chunk_index < batch_size; chunk_index++)
    {
        uncompressed_ptrs[chunk_index] = (char*)data_pointer + chunk_index * chunk_size;
    }
    nvcompBatchedCascadedCompressGetTempSize(batch_size, chunk_size, nvcompBatchedCascadedDefaultOpts, &temp_bytes);
    cudaMalloc(&temp_ptr, temp_bytes);
    size_t max_out_bytes;
    nvcompBatchedCascadedCompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedCascadedDefaultOpts, &max_out_bytes);
    cudaMalloc(&out_ptr, sizeof(size_t)*batch_size);
    cudaMalloc(&out_bytes, sizeof(size_t)*batch_size);
    CHECK_NVCOMP(nvcompBatchedCascadedCompressAsync(uncompressed_ptrs,
                                                    uncompressed_batch,
                                                    chunk_size, batch_size,
                                                    temp_ptr,
                                                    temp_bytes,
                                                    out_ptr,
                                                    out_bytes,
                                                    nvcompBatchedCascadedDefaultOpts,
                                                    0), __FILE__, __LINE__);
}

void GraphicsResource::initialize_nvcomp()
{

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
    //graphicsResource->compress();
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
	log_file.close();
}

