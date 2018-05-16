#!/bin/bash

gfortran \
-c \
photo.f gphoto.f phfit2.f

nvcc \
-Wno-deprecated-gpu-targets \
--device-c \
GwMcmcOnCuda.cu StrctrsAndFnctns.cu

nvcc \
-Wno-deprecated-gpu-targets \
GwMcmcOnCuda.o StrctrsAndFnctns.o \
photo.o gphoto.o phfit2.o \
-lcfitsio -lcusparse -lcublas -lcurand -lcufft \
-o runGwMcmcOnCuda
