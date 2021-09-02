#!/bin/bash

IFN=${1}

FLAGS="-O4 -fPIC"
CPATHS="-I${CUDA_PATH}/include"
LDPATHS="-L${CUDA_PATH}/lib64 -L./"
LIBS="-lOpenCL -lvkfft -lm"

if [[ "fft_interface" == ${IFN} ]]; then
    gcc ${FLAGS} ${CPATHS} ${LDPATHS} fft_interface.cc ${LIBS} -shared -o libvkfft.so
else
    gcc ${FLAGS} ${CPATHS} ${LDPATHS} ${IFN}.cc ${LIBS} -o ${IFN}
fi
