FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG USERNAME=duser
ENV DEBIAN_FRONTEND=noninteractive
ENV WORK_PATH /workspace

# For jp_JP.UTF-8 and JST(Asia/Tokyo)
ENV TZ Asia/Tokyo
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:en
ENV LC_ALL ja_JP.UTF-8
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y language-pack-ja tzdata && \
    rm -rf /var/lib/apt/lists/* && \
    update-locale LANG=ja_JP.UTF-8 LANGUAGE="ja_JP:ja" && \
    echo "${TZ}" > /etc/timezone && \
    rm /etc/localtime && \
    ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# apt-get
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-add-repository ppa:fish-shell/release-3 && \
    apt-add-repository ppa:deadsnakes/ppa && \
		apt-get install -y \
			curl \
			git \
			sudo \
			tree \
			vim \
			wget \
			build-essential \
			unzip \
			libopencv-dev \
			fish \
			fonts-powerline \
			python3.9 \
			python3-pip && \
    apt-get clean && \
		rm -rf /var/lib/apt/lists/* && \
		rm -rf /var/cache/apk/

# python
COPY docker/requirements.txt $WORK_PATH/docker/
ENV PYTHONPATH "/workspace:${PYTHONPATH}"
ENV PIP_OPTIONS "--no-cache-dir --progress-bar off"
RUN pip3 install -U pip && \
    pip3 install ${PIP_OPTIONS} -r $WORK_PATH/docker/requirements.txt && \
		pip3 install ${PIP_OPTIONS} -U setuptools

# nvidia
# WORKDIR /tmp/apex/
# RUN git clone https://github.com/NVIDIA/apex
# WORKDIR /tmp/apex/apex/
# RUN pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# duser setting
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID $USERNAME && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $USERNAME

#dotfiles
ENV DOTFILES_PATH /home/${USERNAME}/dotfiles
RUN git clone https://github.com/tomoino/dotfiles.git $DOTFILES_PATH && \
    chown -R ${USERNAME}:${USERNAME} $DOTFILES_PATH && \
    sudo sh $DOTFILES_PATH/init_docker.sh

WORKDIR /workspace