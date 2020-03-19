cmake_minimum_required( VERSION 2.8 )

project( Cuda_Animal_Detection LANGUAGES CXX CUDA)

## Check if it's windows, if it is, set the OpenCV Directory.
if( WIN32 )
    set( "OpenCV_DIR" "C:/Program Files/opencv/build/x64/vc15/lib" )
endif()

## Find the opencv library.
find_package( OpenCV REQUIRED )

## Find the CUDA library.
find_package( CUDA REQUIRED )

## Use at least C++14.
set (CMAKE_CXX_STANDARD 14)

## Make sure pthread is being used.
Set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pthread")

## Setup the build directory.
include_directories( ${OpenCV_INCLUDE_DIRS} include )
set( EXECUTABLE_OUTPUT_PATH build )
add_executable( Cuda_Animal_Detection ./src/main.cpp ./src/open_image.cpp ./src/mat.cpp ./src/mlp.cpp ./src/ops_cpu.cpp ./src/ops_gpu.cu ./src/ops_hybrid.cu ./src/cnn.cpp ./src/image_handler.cpp)

target_link_libraries( Cuda_Animal_Detection ${OpenCV_LIBS} )

