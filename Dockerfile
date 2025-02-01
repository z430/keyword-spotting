FROM  nvcr.io/nvidia/pytorch:24.12-py3
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN mkdir app
WORKDIR /app
ENV TZ=Asia/Jakarta
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install -y --allow-downgrades --allow-change-held-packages  \
    --no-install-recommends ca-certificates libsm6 libxext6 curl \
    'libsm6' 'libxext6' git build-essential cmake pkg-config unzip yasm \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app
RUN pip install --upgrade -r requirements.txt

WORKDIR /media/ssd/workspace/keyword-spotting
