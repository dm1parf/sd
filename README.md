# New Image Data Transmission Pipeline Tests

## VAE

### None (don't use autoencoder)

use_autoencoder=0
<br>
autoencoder_type=
<br>
config_path=
<br>
ckpt_path=

### VQ-f16

use_autoencoder=1
<br>
autoencoder_type=AutoencoderVQ_F16
<br>
config_path=dependence/config/vq-f16.yaml
<br>
ckpt_path=dependence/ckpt/vq-f16.ckpt

https://github.com/CompVis/stable-diffusion

### Optimized VQ-f16 (torch.trace)

use_autoencoder=1
<br>
autoencoder_type=AutoencoderVQ_F16_Optimized
<br>
config_path=dependence/config/vq-f16.yaml
<br>
ckpt_path=dependence/ckpt/vq-f16.ckpt

https://github.com/CompVis/stable-diffusion

### KL-f16

use_autoencoder=1
<br>
autoencoder_type=AutoencoderKL_F16
<br>
config_path=dependence/config/kl-f16.yaml
<br>
ckpt_path=dependence/ckpt/kl-f16.ckpt

https://github.com/CompVis/stable-diffusion

### KL-f32

use_autoencoder=1
<br>
autoencoder_type=AutoencoderKL_F32
<br>
config_path=dependence/config/kl-f32.yaml
<br>
ckpt_path=dependence/ckpt/kl-f32.ckpt

https://github.com/CompVis/stable-diffusion

### CDC

use_autoencoder=1
<br>
autoencoder_type=AutoencoderCDC
<br>
config_path=
<br>
ckpt_path=dependence/ckpt/cdc1.ckpt

https://github.com/buggyyang/CDC_compression

## SR

### Dummy (cv2.resize)

sr_type=SRDummy
<br>
config_path=
<br>
ckpt_path=

### Real-ESRGAN x2 Plus

sr_type=SRRealESRGAN_x2plus
<br>
config_path=
<br>
ckpt_path=dependence/ckpt/RealESRGAN_x2plus.pth

https://github.com/xinntao/Real-ESRGAN

### APISR: RRDB x2

sr_type=SRAPISR_RRDB_x2
<br>
config_path=
<br>
ckpt_path=dependence/ckpt/2x_APISR_RRDB_GAN_generator.pth

https://github.com/Kiteretsu77/APISR/tree/main

### APISR: RRDB x2 Optimized (torch.trace)

sr_type=SRAPISR_RRDB_x2_Optimized
<br>
config_path=
<br>
ckpt_path=dependence/ckpt/2x_APISR_RRDB_GAN_generator.pth

https://github.com/Kiteretsu77/APISR/tree/main

### APISR: GRL x4

sr_type=SRAPISR_GRL_x4
<br>
config_path=
<br>
ckpt_path=dependence/ckpt/4x_APISR_GRL_GAN_generator.pth

https://github.com/Kiteretsu77/APISR/tree/main
