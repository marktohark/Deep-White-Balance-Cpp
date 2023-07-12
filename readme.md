# Deep White Balance Cpp
## Introduction
C++ implementation of [Deep_White_Balance](https://github.com/mahmoudnafifi/Deep_White_Balance)  
You need to convert the model from [Deep_White_Balance](https://github.com/mahmoudnafifi/Deep_White_Balance)   to ONNX format, or you can download the converted model from the release.  
I have only converted the net_awb.pth model, you will need to convert the others yourself.


## Dependency  
* [xtensor](https://github.com/xtensor-stack/xtensor)
* [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas)
* [opencv 4.5.0](https://github.com/opencv/opencv)
* [onnxruntime](https://github.com/microsoft/onnxruntime)
* blas
* lapack

## build example
P.S. models/awb.onnx is a empty file, you should download from release and cover it.
```shell
mkdir build
cd build
cmake ..
make
./deep_white_balance
```