cmake_minimum_required(VERSION 3.22)
project(deep_white_balance)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR /home/jeremy/Desktop/opencv/build)
find_package(OpenCV REQUIRED)

find_package(xtensor REQUIRED)

add_executable(deep_white_balance main.cpp Awb.cpp Awb.h XTensorHelper.h)

target_link_libraries(deep_white_balance ${OpenCV_LIBS})
target_link_libraries(deep_white_balance blas lapack onnxruntime xtensor)