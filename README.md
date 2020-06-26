[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)

# TensorRT Open Source Software

This repository contains the Open Source Software (OSS) components of NVIDIA TensorRT. Included are the sources for TensorRT plugins and parsers (Caffe and ONNX), as well as sample applications demonstrating usage and capabilities of the TensorRT platform.

# TensorRT PReLU plugin enabled version

**pre-requisite**
You need to prepare your own [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) `.caffemodel` and `deploy.prototext`.

1. ## Generate the TensorRT build container.
   
   **Example: Ubuntu 18.04 with cuda-10.2 for PReLU plugin**

   ```bash
   docker build -f docker/custom.Dockerfile --tag=tensorrt-prelu .
   ```

2. ## Launch the TensorRT build container.

   ```bash
   docker run --rm -ti -u $(id -u):$(id -g) -v $(pwd):/workspace -v <model_dir>:/model -w /workspace tensorrt-prelu
   ```

3. ## Building The TensorRT OSS Components

   (in the container)
   ```bash
   mkdir build && cd build
   cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_BIN_DIR=`pwd`/out
   make -j$(nproc)
   ```

   *Warning: Building OSS components require large memories. Please adjust building thread numbers when you meet `out of memory` error during make.*

4. ## Testing

   ```bash
    LD_PRELOAD="out/libnvinfer_plugin.so.7.0.0:out/libnvcaffeparser.so.7.0.0" out/trtexec --deploy=/model/pose_deploy.prototxt --model=/model/pose_iter_584000.caffemodel --output=net_output --batch=1 --saveEngine=<plan path> <--fp16>
   ```

## Useful Resources

#### TensorRT

* [TensorRT OSS Component Guide](https://github.com/NVIDIA/TensorRT)
* [TensorRT Homepage](https://developer.nvidia.com/tensorrt)
* [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)
* [TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html)
* [TensorRT Discussion Forums](https://devtalk.nvidia.com/default/board/304/tensorrt/)


## Known Issues

#### TensorRT 7.0
* See [Release Notes](https://docs.nvidia.com/deeplearning/sdk/tensorrt-release-notes/tensorrt-7.html#tensorrt-7).
