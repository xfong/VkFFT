#!/bin/bash

IFN=${1}

FLAGS="-O4 -fPIC -Wall"
CPATHS="-I${CUDA_PATH}/include -I../../../vkFFT -std=c99"
LDPATHS="-L${CUDA_PATH}/lib64"
LIBS="-lOpenCL -lm"

gcc ${FLAGS} ${CPATHS} ${LDPATHS} ${IFN}.cc ${LIBS} -o ${IFN}
