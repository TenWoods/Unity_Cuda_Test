#include "cuda_interop.h"
#include <iostream>

UNITY_INTERFACE_EXPORT void SendTextureIDToCuda(int texture_id, int width, int height)
{
	if (graphicsResource == NULL)
	{
		graphicsResource = new GraphicsResource(texture_id, width, height);
	}
}

void GraphicsResource::registerTexture()
{
	CHECK_ERROR(cudaGraphicsGLRegisterImage(&resource, id, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
}

void GraphicsResource::mapResource()
{
	CHECK_ERROR(cudaGraphicsMapResources(1, &resource, 0));
}

void GraphicsResource::copyCudaArray()
{
	CHECK_ERROR(cudaMalloc(&data_pointer, data_length));
	CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
	CHECK_ERROR(cudaMemcpy2DFromArray(data_pointer, width * sizeof(uchar4), array, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice));
	//Debug
	//output_for_debug();
}

void GraphicsResource::unmapResource()
{
	CHECK_ERROR(cudaGraphicsUnmapResources(1, &resource, 0));
}

void GraphicsResource::unregisterResource()
{
	CHECK_ERROR(cudaGraphicsUnregisterResource(resource));
}

void GraphicsResource::output_for_debug()
{
	if (!isFirstDebug)
		return;
	isFirstDebug = false;
	std::ofstream file;
	file.open("debug.ppm");
	void* test = malloc(data_length);
	CHECK_ERROR(cudaMemcpy(test, data_pointer, data_length, cudaMemcpyDeviceToHost));
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

static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
	graphicsResource->registerTexture();
	graphicsResource->mapResource();
	graphicsResource->copyCudaArray();
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

