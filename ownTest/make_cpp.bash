#!/bin/bash

IFN=${1}

FLAGS="-O4 -fPIC"
CPATHS="-I${CUDA_PATH}/include"
LDPATHS="-L${CUDA_PATH}/lib64"
LIBS="-lOpenCL -lm"

g++ ${FLAGS} ${CPATHS} ${LDPATHS} ${IFN}.cpp ${LIBS} -o ${IFN}
