#!/bin/bash

FLAGS="-O4 -fPIC -Wall -DVKFFT_BACKEND=3"
CPATHS="-I${CUDA_PATH}/include"
LDPATHS="-L${CUDA_PATH}/lib64"
LIBS="-lOpenCL -lm"

if [ -f ./vkfft.o ]; then
    rm -f vkfft.o
fi
if [ -f ./libvkfft.so ]; then
    rm -f libvkfft.so
fi
if [ -f ./libvkfft.a ]; then
    rm -f libvkfft.a
fi
if [ -f ../lib/libvkfft.so ]; then
    rm -f ../lib/libvkfft.so
fi
if [ -f ../lib//libvkfft.a ]; then
    rm -f ../lib/libvkfft.a
fi

gcc ${FLAGS} ${CPATHS} ${LDPATHS} -c vkFFT.cc ${LIBS} -o vkfft.o

if [ -f vkfft.o ]; then
    ar -rcs libvkfft.a vkfft.o
    gcc ${FLAGS} ${LDPATHS} vkfft.o ${LIBS} -shared -o libvkfft.so
fi
