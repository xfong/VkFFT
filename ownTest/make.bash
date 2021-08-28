#!/bin/bash

IFN=${1}

FLAGS="-O4 -fPIC"
CPATHS="-I${CUDA_PATH}/include"
LDPATHS="-L${CUDA_PATH}/lib64"
LIBS="-lOpenCL -lm"

gcc ${FLAGS} ${CPATHS} ${LDPATHS}/lib64 ${IFN}.cc ${LIBS} -o ${IFN}
