LIBRARY_PATH := ../lib
OBJ_PATH := obj
CUDA_PATH := /usr
CFLAGS := -Invcomp -IPluginAPI 
LDFLAGS := -L../lib -L$(CUDA_PATH)/lib

$(OBJ_PATH)/Cuda_Interop.o: cuda_interop.cu cuda_interop.h
	nvcc -Xcompiler -fPIC $(CFLAGS) -c cuda_interop.cu -o $(OBJ_PATH)/Cuda_Interop.o

$(LIBRARY_PATH)/libCuda_Interop.so: $(OBJ_PATH)/Cuda_Interop.o
	g++ -shared -fPIC -o ${LIBRARY_PATH}/libCuda_Interop.so ${OBJ_PATH}/Cuda_Interop.o $(LDFLAGS) -lcudart -lnvcomp
	
all : mkdir $(LIBRARY_PATH)/libCuda_Interop.so
mkdir:
	mkdir -p obj
clean:
	rm obj/*
