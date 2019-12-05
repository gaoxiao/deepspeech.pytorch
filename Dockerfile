FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

WORKDIR /workspace/

# install basics
RUN apt-get update -y
RUN apt-get install -y git curl ca-certificates bzip2 cmake tree htop bmon iotop sox libsox-dev libsox-fmt-all vim

# install python deps
RUN pip install cython visdom cffi tensorboardX wget

# install pytorch audio
RUN pip install torchaudio -f https://download.pytorch.org/whl/torch_stable.html

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# install apex
RUN git clone --recursive https://github.com/NVIDIA/apex.git
RUN cd apex; pip install .

# install deepspeech.pytorch
ADD . /workspace/deepspeech.pytorch
RUN cd deepspeech.pytorch; pip install -r requirements.txt
