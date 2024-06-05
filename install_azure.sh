# Install.md

## 1. Install gcc-13 for PyTorch AVX512 optimizations

```bash
sudo mv /etc/apt/sources.list.d/nvidia-docker.list /etc/apt
sudo apt update
sudo apt full-upgrade

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-13 g++-13
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 200
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 200
sudo update-alternatives --display gcc
sudo update-alternatives --display g++
```

## 2.1 Install nvidia-driver-550

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb # 20.04
sudo apt install ./cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install nvidia-driver-550
sudo reboot
rm cuda-keyring_1.1-1_all.deb
```

## 2.2 Install CUDA 12.4

```bash
sudo apt install cuda-toolkit-12-4
sudo bash -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf"
sudo vim /etc/environment # on a new line, type LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo reboot
```

## 2.3 Install cuDNN 9

```bash
sudo apt install cudnn9-cuda-12 libcudnn9-samples
```

## 2.4 Install miscellaneous NVIDIA libraries

```bash
sudo apt install libcusparse-12-4 libcusparse-dev-12-4 libcusparselt0 libcusparselt-dev # CUSparse and CUSparse-LT
sudo apt install nvidia-fabricmanager-550 nvidia-fabricmanager-dev-550 cuda-drivers-fabricmanager-550 # NVIDIA Fabric Manager for NVLink/NVSwitch
sudo apt install libnccl2 libnccl-dev # NCCL
sudo apt install libxnvctrl0=550.* nvidia-settings=550.* # Miscellaneous NVIDIA management tools
sudo apt install nvidia-container-toolkit # 2.5 (Optional) NVIDIA Docker
```

## 2.5 apt clean up

```bash
sudo apt autoremove
sudo apt clean
sudo apt autoclean
```

## 3.1. Install NVIDIA Video Codec SDK headers

```bash
cd
git clone --single-branch --branch n12.0.16.1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git # NOTE: n11.1.5.3 works. n12.0.16.0 works. n12.0.16.1 works. n12.1.14.0 and above do not work.
cd nv-codec-headers
sudo make install
cd
rm -rf nv-codec-headers
```

## 3.2. Install ffmpeg5 with NVIDIA Video Codec SDK support

```bash
cd
git clone --single-branch --branch n5.1.4 https://git.ffmpeg.org/ffmpeg.git # NOTE: 6.0 and above will break decord.
cd ffmpeg
sudo apt install yasm libgnutls28-dev libx264-dev
# NOTE: --enable-nvdec could fail for gpus without nvdec support, such as A100. Works for V100. See https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
export MY_SM=70 # NOTE: For V100, it's 70. See https://developer.nvidia.com/cuda-gpus
./configure \
  --extra-cflags='-I/usr/local/cuda/include -I/usr/local/include' \
  --extra-ldflags='-L/usr/local/cuda/lib64' \
  --nvccflags="-gencode arch=compute_${MY_SM},code=sm_${MY_SM} -O2" \
  --disable-doc \
  --enable-decoder=aac \
  --enable-decoder=h264 \
  --enable-decoder=h264_cuvid \
  --enable-decoder=rawvideo \
  --enable-indev=lavfi \
  --enable-encoder=libx264 \
  --enable-encoder=h264_nvenc \
  --enable-demuxer=mov \
  --enable-muxer=mp4 \
  --enable-filter=scale \
  --enable-filter=testsrc2 \
  --enable-protocol=file \
  --enable-protocol=https \
  --enable-gnutls \
  --enable-gpl \
  --enable-nonfree \
  --enable-cuda \
  --enable-cuda-nvcc \
  --enable-libx264 \
  --enable-nvenc \
  --enable-nvdec \
  --enable-libnpp \
  --enable-cuvid \
  --disable-postproc \
  --enable-shared \
  --enable-static
make clean
make -j
sudo make install
sudo sh -c "echo '/usr/local/lib' >> /etc/ld.so.conf"
sudo ldconfig
cd
rm -rf ffmpeg
```

## 3.3. Confirm your ffmpeg has nvcodec enabled

```bash
# Examples in https://pytorch.org/audio/stable/build.ffmpeg.html#checking-the-intallation
ffprobe -hide_banner -decoders | grep h264
ffmpeg -hide_banner -encoders | grep 264
src="https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
ffmpeg -hide_banner -y -vsync 0 \
     -hwaccel cuvid \
     -hwaccel_output_format cuda \
     -c:v h264_cuvid \
     -resize 360x240 \
     -i "${src}" \
     -c:a copy \
     -c:v h264_nvenc \
     -b:v 5M test.mp4
rm test.mp4
```

## 4.1 Install Anaconda under ${HOME}. Will not work under a network disk

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh # Say yes to everything
source .bashrc
```

## 4.2 Build PyTorch from source

```bash
mv .condarc .condarc-old
conda create -n clean_pytorch_ffmpeg_build cmake ninja intel::mkl-static intel::mkl-include astunparse "expecttest!=0.2.0" hypothesis numpy psutil pyyaml requests setuptools "typing-extensions>=4.8.0" sympy filelock networkx jinja2 fsspec
conda activate clean_pytorch_ffmpeg_build
conda install libgcc-ng=13.1 libstdcxx-ng=13.1 libgcc-ng=13.1 libgomp=13.1 -c defaults -c conda-forge
conda install -c pytorch magma-cuda124
pip install types-dataclasses "optree>=0.9.1" lark
```

```bash
cd && git clone --recursive --single-branch --branch v2.3.0 https://github.com/pytorch/pytorch.git
git submodule sync
git submodule update --init --recursive
# TODO: Monkey-patch ${HOME}/pytorch/aten/src/ATen/core/boxing/impl/boxing.h according to <https://github.com/pytorch/pytorch/issues/122169#issuecomment-2146155541>
```

```bash
export TORCH_CUDA_ARCH_LIST="7.0" # NOTE: For V100, it's 7.0. See https://developer.nvidia.com/cuda-gpus
export USE_FFMPEG=1
export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_SYSTEM_NCCL=1
rm ${CONDA_PREFIX}/lib/libffi.7.so ${CONDA_PREFIX}/lib/libffi.so.7
```

```bash
python setup.py clean && echo "Done Cleaning"
python setup.py install | tee install_pytorch.log # Wait 10 mins for it to finish.
echo "DONE building pytorch" && cd # NOTE it's important to move out of the pytorch build directory to import torch.
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6
```

```bash
python -c "import torch; print(torch.cuda.is_available()); exit()"
python
import torch
torch.rand(2, 3, device='cuda') @ torch.rand(3, 2, device='cuda') # Check CUDA is working
torch.svd(torch.rand(3,3, device='cuda')) # Check MAGMA-CUDA is working
exit() # Get out of the Python shell.
```

## 4.3 Install torchvision

```bash
cd && git clone --recursive --single-branch --branch v0.18.0 https://github.com/pytorch/vision.git
# <!-- sudo ln /usr/local/include/ffnvcodec/dynlink_cuviddec.h /usr/local/include/ffnvcodec/cuviddec.h
# sudo ln /usr/local/include/ffnvcodec/dynlink_nvcuvid.h /usr/local/include/ffnvcodec/nvcuvid.h -->
sudo ln /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
# <!-- export TORCHVISION_INCLUDE=/usr/local/include/ffnvcodec # for cuviddec.h and nvcuvid.h -->
# TORCHVISION_INCLUDE=/usr/local/include/ffnvcodec # for cuviddec.h and nvcuvid.h -->
cd vision
export TORCH_CUDA_ARCH_LIST="7.0" # NOTE: For V100, it's 7.0. See https://developer.nvidia.com/cuda-gpus
export TORCHVISION_INCLUDE=/usr/local/include:/usr/local/include/ffnvcodec:/usr/local/cuda/include # for cuviddec.h and nvcuvid.h
export TORCHVISION_LIBRARY=/usr/local/lib:/usr/lib/x86_64-linux-gnu # for libnvcuvid.so
export USE_FFMPEG=1
export _GLIBCXX_USE_CXX11_ABI=1
python setup.py install
```

# Install torchaudio


# Install project dependencies
pip install -U tqdm gradio matplotlib sentencepiece protobuf transformers tokenizers huggingface_hub accelerate
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release # Change 86 (3090 Ti) to different compute capabillites like before.
pip install -U tqdm gradio matplotlib sentencepiece protobuf transformers tokenizers huggingface_hub accelerate

pip install .


# To run GPT4V
pip install yacs azure-storage-blob ipywidgets


# 3. Install decord
cd
git clone --recursive https://github.com/zhanwenchen/decord
cd decord
pip install .
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
make -j

# Install decord Python bindings
conda activate vtom
pip install .
python setup.py install --user
# Test decord installation
cd examples
# Run all the Jupyter Notebooks under the vtom environment
# You need to install ALSA (`sudo apt install libasound2-dev` and then `pip install simpleaudio opencv-python-headless`)

Additionally, install [FlashAttention](https://github.com/HazyResearch/flash-attention) for training,
```shell
pip install ninja einops
conda activate vtom
cd ~/vtom

cd
git clone --single-branch --branch v2.3.3 git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=4 python setup.py install # Cannot use pip install . on this repo. Also need to specify
conda activate vtom
cd ~/vtom
