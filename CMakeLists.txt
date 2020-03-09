cmake_minimum_required( VERSION 2.8 )

project( Cuda_Animal_Detection )

## Check if it's windows, if it is, set the OpenCV Directory.
##if( WIN32 )
##    set( "OpenCV_DIR" "D:/your/opencv/path" )
##endif()

## Find the opencv library.
find_package( OpenCV REQUIRED )

## Find the CUDA library.
find_package( CUDA REQUIRED )

## Use at least C++11.
set (CMAKE_CXX_STANDARD 11)

## Make sure pthread is being used.
Set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pthread")

## Setup the build directory.
include_directories( ${OpenCV_INCLUDE_DIRS} include )
set( EXECUTABLE_OUTPUT_PATH build )
add_executable( Cuda_Animal_Detection ./src/main.cpp )

target_link_libraries( Cuda_Animal_Detection ${OpenCV_LIBS} )
