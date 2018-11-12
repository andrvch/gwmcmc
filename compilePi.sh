#!/bin/bash

nvcc \
-Wno-deprecated-gpu-targets \
pisimple.cu \
-lcublas -lcurand \
-o runpi
