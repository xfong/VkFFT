#!/bin/bash

IFN=${1}

FLAGS="-O4 -fPIC -Wall -D__DEBUG__=0"
CPATHS="-I${CUDA_PATH}/include -I../../include"
LDPATHS="-L${CUDA_PATH}/lib64 -L../../lib"
LIBS="-lOpenCL -lvkfft -lm"

gcc ${FLAGS} ${CPATHS} ${LDPATHS} ${IFN}.cc ${LIBS} -o ${IFN}
