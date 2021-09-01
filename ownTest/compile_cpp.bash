#!/bin/bash

IFN=${1}

FLAGS="-O4 -fPIC"
CPATHS="-I${CUDA_PATH}/include"
LDPATHS="-L${CUDA_PATH}/lib64"
LIBS="-lOpenCL -lm"

if [ -f ${IFN}.o ]; then
    rm ${IFN}.o
fi
g++ ${FLAGS} ${CPATHS} ${LDPATHS} -c ${IFN}.cpp ${LIBS} -o ${IFN}.o
