#include "cuda_interop.h"
#include <iostream>
#include <fstream>

DllExport int Test(int input)
{
	return input;
}

GraphicsResource* getResource(int texture_id)
{
	if (graphicsResources.find(texture_id) == graphicsResources.end())
	{
		GraphicsResource* resource = new GraphicsResource();
		resource->id = texture_id;
		cudaError err;
		err = cudaGraphicsGLRegisterImage(&resource->resource, texture_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
		if (err != cudaSuccess)
		{
			std::ofstream log_file;
			log_file.open("log.txt");
			log_file << cudaGetErrorName(err) << ': ' << cudaGetErrorString(err) << std::endl;
		}
		graphicsResources[texture_id] = resource;
	}
	return graphicsResources[texture_id];
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

DllExport void Dispose()
{
	for (std::pair<int, GraphicsResource*> p : graphicsResources)
	{
		cudaGraphicsUnmapResources(1, &p.second->resource, 0);
	}
}

