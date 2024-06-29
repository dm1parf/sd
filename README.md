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

<hr>

## <a name="as">1.</a> AS — Artifact Suppressor — Подавитель артефактов
Отвечает не только за собственно подавление артефактов, но и за преобразования torch.Tensor -> np.ndarray и обратно.
Можно убрать автоматическое изменение размера на 512x512, если передать в последние параметры новый промежуточный размер
Если исходные, то изменения размера не происходит вовсе. В качестве примера:<br>
as_params=1280 720

### ASDummy (don't use an artifact suppressor)

as_type=ASDummy
<br>
as_params=

### Cut Edge Colors
Примечание: в as_params указывать параметр delta. Чем выше, тем меньше артефактов, но больше цветовое искажение.

as_type=ASCutEdgeColors
<br>
as_params=15

### Move Distribution
Примечание: техника встречается, например, в примерах к Stable Diffusion и в CDC.

as_type=ASMoveDistribution
<br>
as_params=

### Composit
Примечание: совмещение Cut Edge Colors и Move Distribution.

as_type=ASComposit
<br>
as_params=15

<hr>

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

<hr>

## <a name="quantizer">3.</a> Quantizer — Квантовальщик

### По предварительному квантованию

В качестве параметров можно использовать т.н. предварительное квантование float32 -> float32 для уменьшения числа значимых цифр. Таким образом возможно обеспечить дополнительное сжатие ценой небольшого снижения качества.
<br>
Возможные методы:
- scale
- bitgroom
- granularbr
- bitround

Требуется также параметр nsd (число значимых разрядов?): от 0.
<br>
Параметры должны добавляться в конец. Пример следующий:

quantizer_params=scale 0

### None — Не использовать квантование

use_quantizer=1
<br>
quantizer_type=WorkerQuantLinear
<br>
quantizer_params=

### Linear — Линейное

use_quantizer=1
<br>
quantizer_type=WorkerQuantLinear
<br>
quantizer_params=

А. А. Березкин, А. А. Ченский. Исследование методов квантования при сжатии видеопотока для управления беспилотными системами от первого лица // АПИНО 2024 : Материалы конференции.
<br>
А. А. Березкин, А. А. Ченский. Квантование видеопотока при FPV-управлении беспилотными системами в гибридных сетях связи // 79-я  научно-техническая  конференция  СПб  НТО  РЭС  им. А.С. Попова : Материалы конференции.

### Power — Степенное

use_quantizer=1
<br>
quantizer_type=WorkerQuantPower
<br>
quantizer_params=

А. А. Березкин, А. А. Ченский. Исследование методов квантования при сжатии видеопотока для управления беспилотными системами от первого лица // АПИНО 2024 : Материалы конференции.
<br>
А. А. Березкин, А. А. Ченский. Квантование видеопотока при FPV-управлении беспилотными системами в гибридных сетях связи // 79-я  научно-техническая  конференция  СПб  НТО  РЭС  им. А.С. Попова : Материалы конференции.

### Logistics — Логистическое

use_quantizer=1
<br>
quantizer_type=WorkerQuantLogistics
<br>
quantizer_params=

### Modified Logistics — Модифицированное логистическое

use_quantizer=1
<br>
quantizer_type=WorkerQuantModLogistics
<br>
quantizer_params=

А. А. Березкин, А. А. Ченский. Исследование методов квантования при сжатии видеопотока для управления беспилотными системами от первого лица // АПИНО 2024 : Материалы конференции.
<br>
А. А. Березкин, А. А. Ченский. Квантование видеопотока при FPV-управлении беспилотными системами в гибридных сетях связи // 79-я  научно-техническая  конференция  СПб  НТО  РЭС  им. А.С. Попова : Материалы конференции.

### Odd Power — Нечётностепенное

Примечание: принцип работы полностью отличается от степенного (Power).
<br>
Параметр: power (нечётные натуральные числа).

use_quantizer=1
<br>
quantizer_type=WorkerQuantOddPower
<br>
quantizer_params=3

### Tanh — Гиперболическотангенциальное

use_quantizer=1
<br>
quantizer_type=WorkerQuantTanh
<br>
quantizer_params=

### Modified Tanh — Модифицированно гиперболическотангенциальное

use_quantizer=1
<br>
quantizer_type=WorkerQuantMinTanh
<br>
quantizer_params=

### Double Logistics — Двойное логистическое

use_quantizer=1
<br>
quantizer_type=WorkerQuantDoubleLogistics
<br>
quantizer_params=

### Modified Double Logistics — Модифицированное двойное логистическое

use_quantizer=1
<br>
quantizer_type=WorkerQuantMinDoubleLogistics
<br>
quantizer_params=

### Sinh — Гиперболическосинусоидальное

use_quantizer=1
<br>
quantizer_type=WorkerQuantMinTanh
<br>
quantizer_params=

<hr>

## <a name="compressor">4.</a> Compressor — Компрессор

### Dummy (don't use a compressor)

compressor_type=CompressorDummy
<br>
compressor_params=

### Deflated

compressor_type=CompressorDeflated
<br>
compressor_params=

https://docs.python.org/3/library/zlib.html

### LZMA

compressor_type=CompressorLzma
<br>
compressor_params=

https://docs.python.org/3/library/lzma.html

### GZIP

compressor_type=CompressorGzip
<br>
compressor_params=

https://docs.python.org/3/library/gzip.html

### BZIP2

compressor_type=CompressorBzip2
<br>
compressor_params=

https://docs.python.org/3/library/bz2.html

### ZSTD (ZStandard)

compressor_type=CompressorZstd
<br>
compressor_params=

### Brotli

compressor_type=CompressorBrotli
<br>
compressor_params=

### LZ4

compressor_type=CompressorLz4
<br>
compressor_params=

### LZ4F

compressor_type=CompressorLz4f
<br>
compressor_params=

### LZ4H5

compressor_type=CompressorLz4h5
<br>
compressor_params=

### LZW

compressor_type=CompressorLzw
<br>
compressor_params=

### LZF

compressor_type=CompressorLzf
<br>
compressor_params=

### LZFSE

compressor_type=CompressorLzfse
<br>
compressor_params=

### AEC

compressor_type=CompressorAec
<br>
compressor_params=

### H264
Примечание: только без квантовальщика (quantizer) и автокодировщика (autoencoder)!

compressor_type=CompressorH264
<br>
compressor_params=

https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.pyav.html#module-imageio.plugins.pyav

### H265
Примечание: только без квантовальщика (quantizer) и автокодировщика (autoencoder)!

compressor_type=CompressorH265
<br>
compressor_params=

https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.pyav.html#module-imageio.plugins.pyav

### JPEG
Параметр quality (качество сжатия с потерями, от 0 до 100).

compressor_type=CompressorJpeg
<br>
compressor_params=60

https://imageio.readthedocs.io/en/v2.5.0/format_jpeg-pil.html#jpeg-pil
<br>
ISO/IEC 10918-7

### AVIF
Параметр quality (качество сжатия с потерями, от 0 до 100).

compressor_type=CompressorAvif
<br>
compressor_params=60

https://pypi.org/project/pillow-avif-plugin/
Функционал из libavif.

### HEIC
Параметр quality (качество сжатия с потерями, от 0 до 100).

compressor_type=CompressorHeic
<br>
compressor_params=60

https://pypi.org/project/pillow-avif-plugin/
Функционал из libheif.

### WebP
Первый параметр — lossless (использовать ли сжатие без потерь, 0 или 1), второй — quality (качество сжатия с потерями, от 0 до 100).

compressor_type=CompressorWebp
<br>
compressor_params=0 60

https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#webp

### JPEG LS
Примечание: поддерживается только вариант с потерями.
Параметр level -- уровень сжатия. Режим сжатия без потерь с level = 0. Сжатие с потерями с level от 0 до 10.

compressor_type=CompressorJpegLS
<br>
compressor_params=5

### JPEG XR
Параметр quality (качество сжатия с потерями, от 0 до 100).

compressor_type=CompressorJpegXR
<br>
compressor_params=60

### JPEG XL
Параметры: quality (качество сжатия с потерями, от 0 до 100) и effort (жертвовать ли временем ради улучшения сжатия, от 1 до 9).

compressor_type=CompressorJpegXL
<br>
compressor_params=60 9

### QOI
Примечание: только без автокодировщика!

compressor_type=CompressorQoi
<br>
compressor_params=

https://qoiformat.org/qoi-specification.pdf

<hr>

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

<hr>

## <a name="predictor">6.</a> Predictor — Предиктор

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