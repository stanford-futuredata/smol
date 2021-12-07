cmake_minimum_required (VERSION 3.0)
project (VisionPreproc)

set(CMAKE_CXX_STANDARD 17)

# Libraries
find_package(OpenCV REQUIRED)
find_package(JPEG REQUIRED)
find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
## FFmpeg
find_package(ZLIB REQUIRED)
find_package(LibLZMA REQUIRED)
find_library(AVUTIL_LIB NAMES avutil HINTS /lfs/1/ddkang/local/lib)
find_library(AVFILTER_LIB NAMES avfilter HINTS /lfs/1/ddkang/local/lib)
find_library(AVFORMAT_LIB NAMES avformat HINTS /lfs/1/ddkang/local/lib)
find_library(AVDEVICE_LIB NAMES avdevice HINTS /lfs/1/ddkang/local/lib)
find_library(AVCODEC_LIB NAMES avcodec HINTS /lfs/1/ddkang/local/lib)
find_library(POSTPROC_LIB NAMES postproc HINTS /lfs/1/ddkang/local/lib)
find_library(SWSCALE_LIB NAMES swscale HINTS /lfs/1/ddkang/local/lib)
find_library(SWRESAMPLE_LIB NAMES swresample HINTS /lfs/1/ddkang/local/lib)
## Folly
include(cmake/FindFolly.cmake)
include(cmake/FindGlog.cmake)
include(cmake/FindDoubleConversion.cmake)
## TensorRT
include(cmake/FindTensorRT.cmake)
## ONNX
find_library(ONNX_LIB onnx HINTS /lfs/1/ddkang/local/lib)
find_library(ONNX_PROTO_LIB onnx_proto HINTS /lfs/1/ddkang/local/lib)
find_path(ONNX_INCLUDE onnx HINTS /lfs/1/ddkang/local/include)
## nvinfer
find_library(NVINFER_LIB nvinfer ${TensorRT_LIBRARY})
find_library(NVONNX_LIB nvonnxparser ${TensorRT_LIBRARY})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DONNX_ML=1 -O3")
set(ALL_LIBS
  ${OpenCV_LIBS} ${JPEG_LIBRARIES} ${CUDA_LIBRARIES}
  ${ZLIB_LIBRARIES} ${LIBLZMA_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${AVUTIL_LIB} ${AVFILTER_LIB} ${AVFORMAT_LIB} ${AVCODEC_LIB} ${AVDEVICE_LIB}
  ${POSTPROC_LIB} ${SWSCALE_LIB} ${SWRESAMPLE_LIB}
  ${Protobuf_LIBRARIES} ${FOLLY_LIBRARIES} ${GLOG_LIBRARIES}
  ${DOUBLE_CONVERSION_LIBRARY}
  ${ONNX_LIB} ${ONNX_PROTO_LIB}
  ${NVINFER_LIB} ${NVONNX_LIB}
  "stdc++fs" "x264")



# My code
include_directories("include")
file(GLOB SOURCES "src/*.cc")

# This _must_ be STATIC due to CUDA fuckery
add_library(trt_common STATIC ${SOURCES})
target_link_libraries(trt_common PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(trt_common PUBLIC /usr/local/include/opencv4)

add_executable(runner runner.cc)
target_link_libraries(runner PUBLIC OpenMP::OpenMP_CXX trt_common ${ALL_LIBS})
target_include_directories(runner PUBLIC ${ONNX_INCLUDE})

add_executable(video_runner video_runner.cc)
target_link_libraries(video_runner PUBLIC OpenMP::OpenMP_CXX trt_common ${ALL_LIBS})
target_include_directories(video_runner PUBLIC ${ONNX_INCLUDE})
