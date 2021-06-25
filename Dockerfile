FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y 

RUN pip3 install --upgrade pip
RUN pip3 install numpy scipy sklearn opencv-contrib-python tqdm tensorflow
RUN pip3 install matplotlib

ENV CUDA_VISIBLE_DEVICES=0
ENV TF_FORCE_GPU_ALLOW_GROWTH='true'



