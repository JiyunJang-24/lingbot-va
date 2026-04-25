uv venv .venv --python 3.10.16 --prompt "$(basename "$PWD")"
source .venv/bin/activate
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
uv pip install setuptools
pip install ninja packaging wheel
MAX_JOBS=16 pip install flash-attn --no-build-isolation -v
uv pip install flash-attn --no-build-isolation
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
# sudo cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
# sudo apt-get update
# sudo apt-get -y install cuda

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
# sudo apt-get update
# sudo apt-get -y install cuda-toolkit-12-1

# export PATH=/usr/local/cuda-12.1/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
# source ~/.bashrc



# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
# sudo apt-get update
# sudo apt-get -y install cuda-toolkit-11-8

# export PATH=/usr/local/cuda-11.8/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
# source ~/.bashrc