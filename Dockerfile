FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
# FROM nvcr.io/nvidia/pytorch:22.06-py3
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3-pip \
	python3-setuptools \
	build-essential \
	&& \
	apt-get clean && \
	python -m pip install --upgrade pip

WORKDIR /workspace
COPY ./   /workspace

RUN pip install pip -U
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install -e .

