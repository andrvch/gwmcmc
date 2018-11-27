#!/bin/bash

nvcc \
-Wno-deprecated-gpu-targets \
pisimpleHost.cu \
-lcurand \
-o runpiHost
