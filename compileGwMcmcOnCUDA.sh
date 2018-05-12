#!/bin/bash

gfortran \
-c \
photo.f gphoto.f phfit2.f

nvcc \
-Wno-deprecated-gpu-targets \
--device-c \
GwMcmcOnCuda.cu GwMcmcStructuresFunctionsAndKernels.cu

nvcc \
-Wno-deprecated-gpu-targets \
GwMcmcOnCuda.o GwMcmcStructuresFunctionsAndKernels.o \
photo.o gphoto.o phfit2.o \
-lcfitsio -lcusparse -lcublas -lcurand -lcufft \
-o runGwMcmcOnCuda
