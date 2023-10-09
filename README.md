# sd

exp: 29 - sd_inp 1 step
exp: 32 - sd 3 step
exp: 33 - sd 1 step

Всего в датасете 197975 картинок

# INSTALL
gh repo clone NIRteam/sd
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 

pip install git+https://github.com/huggingface/transformers
pip install accelerate
pip3 install ffmpeg-quality-metrics

<!-- install ffmpeg -->
mkdir build && cd build

<!-- build tools -->
sudo apt-get install ninja-build meson pkg-config

<!-- libvmaf tools -->
sudo apt install nasm

<!-- ffmpeg prerequisites -->
sudo apt install -y libx264-dev libx265-dev gnutls-dev

<!-- INSTALL libvmaf -->
<!-- https://github.com/yash1994/Build-FFmpeg-with-libvmaf -->
<!-- https://github.com/Netflix/vmaf/releases -->
wget https://github.com/Netflix/vmaf/archive/refs/tags/v2.3.1.tar.gz
tar -xf v2.3.1.tar.gz 
cd vmaf-2.3.1/libvmaf
meson build --buildtype release
ninja -vC build
ninja -vC build install

<!-- install ffmpeg with libmvaf -->
wget https://ffmpeg.org/releases/ffmpeg-6.0.tar.xz
tar -xf ffmpeg-6.0.tar.xz 
cd ffmpeg-6.0/
./configure --enable-gpl --enable-libx264 --enable-libx265 --enable-nonfree --enable-libvmaf --enable-version3

sudo make -j 22
sudo make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

<!-- INSTALL sd_inpainting -->
pip install --upgrade diffusers[torch]
