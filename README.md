nvidia-smi 
# gpu 사용량 체크 명령어. 

cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
# 위 명령어로 cuDNN 확인 없다면 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
# 위 명령어로 설치

pip install tensorflow==2.11.0

