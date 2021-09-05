cmake \
  -D JPEG_LIBRARY=/lfs/1/ddkang/local/lib/libjpeg.so \
  -D JPEG_INCLUDE_DIR=/lfs/1/ddkang/local/include/ \
  -D FOLLY_ROOT_DIR=/lfs/1/ddkang/local/ \
  -D TensorRT_LIBRARY=/usr/local/cuda/TensorRT-5.1.5.0/lib \
  -D TensorRT_INCLUDE_DIR=/usr/local/cuda/TensorRT-5.1.5.0/include \
  ..
