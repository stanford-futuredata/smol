sudo apt update
sudo apt upgrade -y
sudo apt install gnupg-curl
echo "INSTALLING CUDA"
mkdir install ; cd install
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64
sudo dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64
sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
sudo apt-get update -y
time sudo apt-get install cuda -y
cd ..
echo "INSTALLING LIBCUDNN and TensorRT"
sudo dpkg -i libcudnn7_7.5.0.56-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.5.0.56-1+cuda10.0_amd64.deb
sudo dpkg -i nv-tensorrt-repo-ubuntu1604-cuda10.0-trt5.1.2.2-rc-20190227_1-1_amd64.deb
sudo apt update -y
sudo apt install tensorrt libnvinfer5 -y
sudo apt-get update -y
echo "INSTALLING PROTOBUF AND ONNX"
sudo apt install libprotobuf-dev protobuf-compiler cmake -y
git clone --recursive https://github.com/onnx/onnx.git
cd onnx
git checkout 5b7ac729f08a04b62a8ef401aa149ba25f07d26c
mkdir build ; cd build
cmake ..
make -j8
sudo make install -j8
cd ../../
echo "INSTALLING YAML CPP"
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
mkdir build; cd build
cmake -DBUILD_SHARED_LIBS=ON ..
make -j8
sudo make install
cd ../../
echo "INSTALLING LIBJPEGTURBO"
sudo dpkg -i libjpeg-turbo-official_2.0.1_amd64.deb
sudo apt-get install -f
echo "INSTALLING OPENCV"
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build; cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local /home/ankitmathur/opencv
make -j7
sudo make install
cd
echo "EDITING BASH PROFILE"
echo "export CUDA_HOME=/usr/local/cuda" >> .bashrc
echo "export DYLD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$DYLD_LIBRARY_PATH" >> .bashrc
echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> .bashrc
echo "export C_INCLUDE_PATH=\$CUDA_HOME/include:\$C_INCLUDE_PATH" >> .bashrc
echo "export CPLUS_INCLUDE_PATH=\$CUDA_HOME/include:\$CPLUS_INCLUDE_PATH" >> .bashrc
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> .bashrc
echo "export LD_RUN_PATH=\$CUDA_HOME/lib64:\$LD_RUN_PATH" >> .bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/libjpeg-turbo/lib64/" >> .bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib" >> .bashrc
source .bashrc
