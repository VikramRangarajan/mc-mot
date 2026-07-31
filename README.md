# Multi Camera Multi-Object-Tracking (MC-MOT)

## Rust build prerequisites

The Rust OpenCV bindings require `libclang` during compilation. Install LLVM/Clang for your platform and configure it locally; machine-specific `.cargo/config.toml` files are intentionally ignored.

macOS/Homebrew:

```bash
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
cargo run --release --manifest-path app/Cargo.toml
```

Linux: install `clang` and `libclang-dev`. Windows: install LLVM and set `LIBCLANG_PATH` to its `bin` directory.

## Install

To install, run:

### 1. Create a new conda env:

```
conda create -n mc-mot python=3.10
```

**NOTE**: If you already have one conda env, just move to the next step.

### 2. Install the dependencies

#### 2.1 PyTorch

```
conda install pytorch torchvision pytorch-cuda=your-cuda-version -c pytorch -c nvidia
```

**Replace** the `your-cuda-version` text by your CUDA version. You can check that with `nvidia-smi`.

**However**, if you don't have a GPU from NVIDIA, you can use the CPU version of PyTorch. Then, you can just install with `pip`:

```
pip install torch torchvision
```

#### 2.2 Other Dependencies

Then, we can just install the other dependencies with:

```
pip install -r requirements.txt
```

## Running

1. We'll download the `cam1.mp4` and `cam4.mp4` video files from the EPFL dataset.

[cam1.mp4](https://drive.google.com/file/d/1sGUnExmJM2_tFuBd9LNlexf0LN2m0_c-/view)

[cam4.mp4](https://drive.google.com/file/d/1sXn70X-bV_YGPv43r4-iMtK_Js09eUVB/view)

Store then in a folder that is easy to access.

2. Run the homography calibration:

```
python3 calibrate.py --video1 /path/to/cam1.mp4 --video2 /path/to/cam4.mp4 --homography-pth /path/to/save/the/homography/matrix
```

You can also do `python3 calibrate.py --help`

At the end, you'll visualize how the bounding boxes are being projected from one camera to the other one.

3. Now, run the `main.py`:

```
python3 main.py --video1 /path/to/cam1.mp4 --video2 /path/to/cam4.mp4 --homography /path/to/homography/matrix.npy
```

For the other parameters, run:

```
python3 main.py --help
```

## Side Notes

This repository was tested with the following specs:

1. NVIDIA GeForce 3060 Laptop GPU (6Gb).
2. i7-11800H 16 Cores.
3. Ubuntu 20.04 (LTS).
4. CudaToolkit 11.8
5. NVIDIA Driver 520.61.05

You should have similar results with similar hardware as above.

The Rust app uses the Apache-2.0 RF-DETR Medium ONNX export through OpenCV DNN.
If you run on a CPU, export/use the RF-DETR Nano or Small variant instead.
