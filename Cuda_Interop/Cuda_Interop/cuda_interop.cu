#include "cuda_interop.h"
#include <iostream>

DllExport int Test(int input)
{
	return input;
}

GraphicsResource* getResource(int texture_id)
{
	GraphicsResource* resource = new GraphicsResource();
	resource->id = texture_id;
	cudaGraphicsGLRegisterImage(&resource->resource, texture_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	return NULL;
}

void readTexture(int texture_id, int width, int height)
{
	GraphicsResource* resource = getResource(texture_id);
	resource->width = width;
	resource->height = height;
}

DllExport void SendTextureToCuda(int texture_id, int width, int height)
{
	readTexture(texture_id, width, height);
}

