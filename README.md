# TensorRT custom plugin

Add some new custom tensorRT plugin

## New plugin

- [BatchedNMSLandmark_TRT](./plugin/batchedNMSLandmarkPlugin/), BatchedNMSLandmarkDynamic_TRT: Batched NMS with face landmark
- [BatchedNMSLandmarkConf_TRT](./plugin/batchedNMSLandmarkConfPlugin/), BatchedNMSLandmarkConfDynamic_TRT: Batched NMS with face lanmdark & confidence
- [EfficientNMSLandmark_TRT](./plugin/efficientNMSLandmarkPlugin/): Efficient NMS with face landmark
- [EfficientNMSCustom_TRT](./plugin/efficientNMSCustomPlugin/): Same Efficient NMS, but return boxes indices
- [RoIAlignDynamic](./plugin/roIAlignPlugin/): Same ONNX RoIAlign, copy from [MMCVRoIAlign](https://github.com/open-mmlab/mmdeploy)
- [RoIAlign2Dynamic](./plugin/roIAlign2Plugin/): Same as pyramidROIAlignPlugin, but only one feature_map.

## Prerequisites

- Deepstream 6.3.0

## Downloading TensorRT Build

1. #### Download TensorRT OSS
```bash
git clone -b master https://gitlab.aigroup.uz/sh.yuldoshov/tensorrt.git TensorRT
cd TensorRT
git submodule update --init --recursive
export TRT_OSSPATH=$(pwd)
```

## Building TensorRT-OSS
* Generate Makefiles and build.

  **Example: Linux (x86-64) build with default cuda-11.8.0**
```bash
export TRT_LIBPATH=/usr/lib/x86_64-linux-gnu/
cd $TRT_OSSPATH
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=pwd/out -DGPU_ARCHS=89 -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DCMAKE_C_COMPILER=/usr/bin/gcc
make nvinfer_plugin -j$(nproc)
```

After building ends successfully, libnvinfer_plugin.so\* will be generated under ./build/out.

### 3. Replace "libnvinfer_plugin.so\*"

```
// backup original libnvinfer_plugin.so.x.y, e.g. libnvinfer_plugin.so.8.0.0
sudo mv /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.p.q ~/libnvinfer_plugin.so.8.p.q.bak
// only replace the real file, don't touch the link files, e.g. libnvinfer_plugin.so, libnvinfer_plugin.so.8
sudo cp $TRT_SOURCE/build/out/libnvinfer_plugin.so.8.m.n  /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.p.q
sudo ldconfig
```
