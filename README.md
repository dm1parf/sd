# New Image Data Transmission Pipeline

## Contents -- Оглавление
[1. AS — Artifact Suppressor — Подавитель артефактов.](#as)
<br>
[2. AE — Autoencoder — Автокодировщик.](#ae)
<br>
[3. Quantizer — Квантовальщик.](#quantizer)
<br>
[4. Compressor — Компрессор.](#compressor)
<br>
[5. SR — Super Resolution — Сверхразрешение.](#sr)
<br>
[6. Predictor — Предиктор.](#predictor)

## <a name="as">1.</a> AS — Artifact Suppressor — Подавитель артефактов

### None (don't use an artifact suppressor)

use_as=0
<br>
as_type=
<br>
as_params=

### Cut Edge Colors
Примечание: в as_params указывать параметр delta. Чем выше, тем меньше артефактов, но больше цветовое искажение.

use_as=1
<br>
as_type=ASCutEdgeColors
<br>
as_params=15

## <a name="ae">2.</a> AE — Autoencoder — Автокодировщик

### None (don't use an autoencoder)

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

## <a name="quantizer">3.</a> Quantizer — Квантовальщик

### None (don't use a quantizer)

use_quantizer=0
<br>
quantizer_type=QuantLogistics

### Linear

use_quantizer=1
<br>
quantizer_type=WorkerQuantLinear

А. А. Березкин, А. А. Ченский. Исследование методов квантования при сжатии видеопотока для управления беспилотными системами от первого лица // АПИНО 2024 : Материалы конференции.
<br>
А. А. Березкин, А. А. Ченский. Квантование видеопотока при FPV-управлении беспилотными системами в гибридных сетях связи // 79-я  научно-техническая  конференция  СПб  НТО  РЭС  им. А.С. Попова : Материалы конференции.

### Power

use_quantizer=1
<br>
quantizer_type=WorkerQuantPower

А. А. Березкин, А. А. Ченский. Исследование методов квантования при сжатии видеопотока для управления беспилотными системами от первого лица // АПИНО 2024 : Материалы конференции.
<br>
А. А. Березкин, А. А. Ченский. Квантование видеопотока при FPV-управлении беспилотными системами в гибридных сетях связи // 79-я  научно-техническая  конференция  СПб  НТО  РЭС  им. А.С. Попова : Материалы конференции.

### Logistics

use_quantizer=1
<br>
quantizer_type=WorkerQuantLogistics

А. А. Березкин, А. А. Ченский. Исследование методов квантования при сжатии видеопотока для управления беспилотными системами от первого лица // АПИНО 2024 : Материалы конференции.
<br>
А. А. Березкин, А. А. Ченский. Квантование видеопотока при FPV-управлении беспилотными системами в гибридных сетях связи // 79-я  научно-техническая  конференция  СПб  НТО  РЭС  им. А.С. Попова : Материалы конференции.

## <a name="compressor">4.</a> Compressor — Компрессор

### Dummy (don't use a compressor)

compressor_type=CompressorDummy

### Deflated

compressor_type=CompressorDeflated

https://docs.python.org/3/library/zlib.html

### LZMA

compressor_type=CompressorLzma

https://docs.python.org/3/library/lzma.html

### GZIP

compressor_type=CompressorGzip

https://docs.python.org/3/library/gzip.html

### BZIP2

compressor_type=CompressorBzip2

https://docs.python.org/3/library/bz2.html

### H264
Примечание: только без квантовальщика (quantizer) и автокодировщика (autoencoder)!

compressor_type=CompressorH264

https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.pyav.html#module-imageio.plugins.pyav

### H265
Примечание: только без квантовальщика (quantizer) и автокодировщика (autoencoder)!

compressor_type=CompressorH265

https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.pyav.html#module-imageio.plugins.pyav

## <a name="sr">5.</a> SR — Super Resolution — Сверхразрешение

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

## <a name="quantizer">6.</a> Predictor — Предиктор

## None (don't use a predictor)

use_predictor=0
<br>
predictor_type=
<br>
config_path=

## Dummy (fake predictor — the same images of the list)

use_predictor=1
<br>
predictor_type=PredictorDummy
<br>
config_path=

## DMVFN

use_predictor=1
<br>
predictor_type=PredictorDMVFN
<br>
config_path=dependence/config/dmvfn_city.pkl

https://github.com/megvii-research/CVPR2023-DMVFN