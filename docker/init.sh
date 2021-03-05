#!/bin/sh
# default runtime が nvidia でない場合、コンテナ内で実行
mkdir /tmp/apex/
cd /tmp/apex/
git clone https://github.com/NVIDIA/apex
cd ./apex/
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .