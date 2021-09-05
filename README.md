# Smol

Code release for [Jointly Optimizing Preprocessing and Inference for DNN-based Visual Analytics](http://vldb.org/pvldb/vol14/p87-kang.pdf). If you find this code useful, please cite us!

```
@article{kang2021jointly,
  title={Jointly optimizing preprocessing and inference for DNN-based visual analytics},
  author={Kang, Daniel and Mathur, Ankit and Veeramacheneni, Teja and Bailis, Peter and Zaharia, Matei},
  journal={PVLDB},
  year={2021}
}
```


## Setup 
Start with a `g4dn.xlarge` AWS instance, using the following AMI: 
`Ubuntu Server 18.04 LTS (HVM), SSD Volume Type`

Set the SSD memory to be around 100GB (you can adjust this based on your specific needs). 


### Step 1: Install CUDA and associated libraries

Follow the instructions here: https://ddkang.github.io/2020/01/02/installing-tensorrt.html


### Step 2: Install extra libraries

You will need to install:
- conda (for the latest CMake)
- FFMpeg
- OpenCV (+ contrib)
- folly
- onnx
- yaml-cpp

You may need to add these lines to your bashrc:
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
export LD_RUN_PATH=$LD_RUN_PATH:/usr/local/lib/
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/lib/
```

For all the `cmake` commands, you can install locally by adding `cmake -DCMAKE_INSTALL_PREFIX=..`

#### conda

```sh
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh

conda create -n dev anaconda
conda install -c anaconda cmake
```

You will need to supervise this install. You can avoid this step by manually installing the latest CMake.

#### FFmpeg

```sh
sudo apt install yasm

git clone https://github.com/FFmpeg/FFmpeg.git
cd FFmpeg
git checkout 4c07985
mkdir build; cd build
../configure --enable-gpl --enable-postproc --enable-pic --enable-shared
make -j$(nproc)
sudo make install -j$(nproc)
```

Add `--prefix=...` to `configure` to install locally.


#### OpenCV

Note: do not run this in the conda environment. You can alternatively install in conda and set the appropriate paths to the conda install.

```sh
sudo apt install build-essential
sudo apt install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
sudo apt install libtiff5-dev
sudo apt install libx264-dev

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
mkdir build ; cd build
cmake -D WITH_FFMPEG=OFF -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
make -j$(nproc)
sudo make install -j$(nproc)
```

#### folly

You need to install fmt from source:
```sh
git clone https://github.com/fmtlib/fmt.git && cd fmt
mkdir _build && cd _build
cmake ..

make -j$(nproc)
sudo make install -j$(nproc)
```

```sh
sudo apt install libgoogle-glog-dev
sudo apt-get install \
    g++ \
    cmake \
    libboost-all-dev \
    libevent-dev \
    libdouble-conversion-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libiberty-dev \
    liblz4-dev \
    liblzma-dev \
    libsnappy-dev \
    make \
    zlib1g-dev \
    binutils-dev \
    libjemalloc-dev \
    libssl-dev \
    pkg-config \
    libunwind-dev
sudo apt-get install \
    libunwind8-dev \
    libelf-dev \
    libdwarf-dev


git clone https://github.com/facebook/folly.git
cd folly
git checkout efaea2394de97a4cbcc6e504ae8eb315a4e4aed4
mkdir _build && cd _build
cmake ..
make -j$(nproc)
sudo make install -j$(nproc)
```

Later versions of `folly` do not work.


#### onnx

Ensure this is executed in the conda environment.

```sh
sudo apt-get update
sudo apt install libprotobuf-dev protobuf-compiler cmake -y

git clone --recursive https://github.com/onnx/onnx.git
cd onnx
mkdir build ; cd build
cmake ..
make -j$(nproc)
sudo make install -j$(nproc)
```

#### yaml-cpp

Ensure this is in the conda environment.

```sh
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
mkdir build ; cd build
cmake -DBUILD_SHARED_LIBS=ON ..
make -j$(nproc)
sudo make install -j$(nproc)
```

#### libjpeg-turbo
Download the following file:
https://sourceforge.net/projects/libjpeg-turbo/files/2.0.3/libjpeg-turbo-official_2.0.3_amd64.deb/download

SCP the deb file to the remote machine.

```sh
sudo dpkg -i libjpeg-turbo-official_2.0.3_amd64.deb
sudo apt-get install -f
```
Add the following to your .bashrc
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/libjpeg-turbo/lib64/
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/opt/libjpeg-turbo/include/
```


### Cloning the code
Clone and build the repository:
```sh
git clone https://github.com/stanford-futuredata/image-serving.git
cd e2e/trt
mkdir build ; cd build
cmake ..
make -j$(nproc)
```

You may need to specify the `yaml-cpp` library and the OpenCV version.

Get the Imagenet val dataset onto your remote machine. Use the following script to organize it into a format that the InferenceServer will be able to understand:
```sh
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
```

## Setting up a Preprocessing Configuration
### Fields
See `im-single-full-base.yaml` in `cfgs` as an example:
- `model-config`: Contains information of the models.
- `experiment-type`: Whether or not to do `full` or `infer-only`.
- `experiment-config`: Other experimental configuration, including loading, inference, writing out predictions, and the multiplier.
- `criterion`: For filtering.
- `infer-config`: Whether or not to do `memcpy` (`do-memcpy`).

The video config is similar.

### Execution
Create a YAML configuration file by filling out the above fields (a sample configuration is included). Run:
```sh
./runner <path to config file>
```
