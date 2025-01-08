import os
import sys
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
import numpy as np
from production_system.production_workers import WorkerASDummy, WorkerCompressorDummy, WorkerSRDummy


class NeuroCodec:
    """Нейросетевой кодек: кодер + декодер."""

    def __init__(self, as_=None, vae=None, quant=None, compressor=None, sr=None):
        if as_:
            self._as = as_
        else:
            self._as = WorkerASDummy()

        self._quant = quant

        if compressor:
            self._compressor = compressor
        else:
            self._compressor = WorkerCompressorDummy()

        self._vae = vae
        if self._quant:
            self.dest_type = np.uint8
        else:
            if self._vae:
                self.dest_type = self._vae.nominal_type
            else:
                self.dest_type = np.float16
        if vae:
            self.dest_shape = self._vae.z_shape
        else:
            self.dest_shape = (1, 3, 512, 512)

        if sr:
            self._sr = sr
        else:
            self._sr = WorkerSRDummy()

    def decode_frame(self, binary, dest_height=720, dest_width=1280):
        """Декодировать сжатое бинарное представление кадра."""

        quant_latent = self._compressor.decompress_work(binary,
                                                        dest_shape=self.dest_shape,
                                                        dest_type=self.dest_type)
        if self._quant:
            latent = self._quant.dequant_work(quant_latent, dest_type=self._vae.nominal_type)
        else:
            latent = quant_latent
        if self._vae:
            image = self._vae.decode_work(latent)
        else:
            image = latent
        frame = self._as.restore_work(image)
        restored_frame = self._sr.sr_work(frame, dest_size=[dest_width, dest_height])

        return restored_frame

    def encode_frame(self, frame):
        """Кодировать кадр в сжатую бинарную последовательность."""

        image = self._as.prepare_work(frame)
        if self._vae:
            latent = self._vae.encode_work(image)
        else:
            latent = image

        if self._quant:
            quant_latent, _ = self._quant.quant_work(latent)
        else:
            quant_latent = latent
        binary = self._compressor.compress_work(quant_latent)

        return binary
