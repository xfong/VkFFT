#!/bin/bash

IFN=$1

FLAGS="-O4 -fPIC -m64"
CPATHS="-I${CUDA_PATH}/include"
LDPATHS="-L${CUDA_PATH}/lib64"
LIBS="-lOpenCL -lm"

if [ -f ${IFN}.o ]; then
    rm ${IFN}.o
fi
gcc ${FLAGS} ${CPATHS} ${LDPATHS}/lib64 -c ${IFN}.cc ${LIBS} -o ${IFN}.o
