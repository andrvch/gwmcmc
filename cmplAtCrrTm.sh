#!/bin/bash

nvcc \
-Wno-deprecated-gpu-targets \
--device-c \
atcrrtm.cu StrctrsAndFnctns.cu

nvcc \
-Wno-deprecated-gpu-targets \
atcrrtm.o StrctrsAndFnctns.o \
-lcfitsio -lcusparse -lcublas -lcurand -lcufft \
-o runAtCrrTm
