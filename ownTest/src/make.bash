#!/bin/bash

FLAGS="-O4 -fPIC -Wall -std=c99 -D__DEBUG__=1"
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

gcc ${FLAGS} ${CPATHS} ${LDPATHS} -c fft_interface.cc ${LIBS} -o fft_interface.o

if [ -f fft_interface.o ]; then
    ar -rcs libvkfft.a fft_interface.o
    gcc ${FLAGS} ${LDPATHS} fft_interface.o ${LIBS} -shared -o libvkfft.so
    cp libvkfft.* ../lib/.
fi
