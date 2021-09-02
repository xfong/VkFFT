#!/bin/bash

IFN=${1}

FLAGS="-O4 -fPIC -Wall"
CPATHS="-I${CUDA_PATH}/include -std=c99"
LDPATHS="-L${CUDA_PATH}/lib64"
LIBS="-lOpenCL -lm"

if [ -f ${IFN}.o ]; then
    rm ${IFN}.o
fi
gcc ${FLAGS} ${CPATHS} ${LDPATHS} -c ${IFN}.cc ${LIBS} -o ${IFN}.o
