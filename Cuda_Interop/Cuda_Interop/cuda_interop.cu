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

void GraphicsResource::mapCudaArray()
{
	CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
}

void GraphicsResource::copyCudaArray()
{

}

static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
	graphicsResource->registerTexture();
	graphicsResource->mapCudaArray();
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

