import os
import sys

cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
import numpy as np
from production_system.production_workers import (WorkerASMoveDistribution, WorkerQuantLinear, WorkerAutoencoderKL_F16,
                                                  WorkerAutoencoderKL_F4, WorkerCompressorJpegXL,
                                                  WorkerCompressorJpegXR,
                                                  WorkerCompressorAvif, WorkerQuantPower)
from production_system.neuro_codec import NeuroCodec


class ConfigurationGuardian:
    onnx_model_path_lib = {
        "AutoencoderKL_F4": ("dependence/onnx/kl-f4_encoder.onnx", "dependence/onnx/kl-f4_decoder.onnx"),
        "AutoencoderKL_F16": ("dependence/onnx/kl-f16_encoder.onnx", "dependence/onnx/kl-f16_decoder.onnx"),
    }

    def __init__(self, max_worksize, enable_encoder=False, enable_decoder=False):
        # Определение конфигураций

        self.enable_encoder = enable_encoder
        self.enable_decoder = enable_decoder

        as_ = WorkerASMoveDistribution()

        kl_f4_coder_path = self.onnx_model_path_lib["AutoencoderKL_F4"][0] if self.enable_encoder else ""
        kl_f4_decoder_path = self.onnx_model_path_lib["AutoencoderKL_F4"][1] if self.enable_decoder else ""
        kl_f16_coder_path = self.onnx_model_path_lib["AutoencoderKL_F16"][0] if self.enable_encoder else ""
        kl_f16_decoder_path = self.onnx_model_path_lib["AutoencoderKL_F16"][1] if self.enable_decoder else ""

        kl_f4 = WorkerAutoencoderKL_F4(encoder_path=kl_f4_coder_path, decoder_path=kl_f4_decoder_path,
                                       max_worksize=max_worksize)
        kl_f16 = WorkerAutoencoderKL_F16(encoder_path=kl_f16_coder_path, decoder_path=kl_f16_decoder_path,
                                         max_worksize=max_worksize)

        quant_lin_bitround1_klf4 = WorkerQuantLinear(pre_quant="bitround", nsd=1)
        quant_lin_bitround1_klf4.adjust_params(autoencoder_worker=kl_f4.nominal_name)
        quant_lin_scale1_klf4 = WorkerQuantLinear(pre_quant="scale", nsd=1)
        quant_lin_scale1_klf4.adjust_params(autoencoder_worker=kl_f4.nominal_name)
        quant_lin_scale1_klf16 = WorkerQuantLinear(pre_quant="scale", nsd=1)
        quant_lin_scale1_klf16.adjust_params(autoencoder_worker=kl_f16.nominal_name)
        quant_lin_klf16 = WorkerQuantLinear()
        quant_lin_klf16.adjust_params(autoencoder_worker=kl_f16.nominal_name)
        quant_pow_scale1_klf16 = WorkerQuantPower(pre_quant="scale", nsd=1)
        quant_pow_scale1_klf16.adjust_params(autoencoder_worker=kl_f16.nominal_name)

        compress_jpegxl65 = WorkerCompressorJpegXL(65)
        compress_jpegxr55 = WorkerCompressorJpegXR(55)
        compress_jpegxr60 = WorkerCompressorJpegXR(60)
        compress_jpegxr65 = WorkerCompressorJpegXR(65)
        compress_jpegxr70 = WorkerCompressorJpegXR(70)
        compress_jpegxr75 = WorkerCompressorJpegXR(75)
        compress_jpegxr80 = WorkerCompressorJpegXR(80)
        compress_jpegxr85 = WorkerCompressorJpegXR(85)
        compress_avif75 = WorkerCompressorAvif(75)
        compress_avif80 = WorkerCompressorAvif(80)

        neuro_cfg1 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_bitround1_klf4,
                                compressor=compress_jpegxl65)
        neuro_cfg2 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr55)
        neuro_cfg3 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr60)
        neuro_cfg4 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr65)
        neuro_cfg5 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr70)
        neuro_cfg6 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr75)
        neuro_cfg7 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr80)
        neuro_cfg8 = NeuroCodec(as_=as_,
                                vae=kl_f4,
                                quant=quant_lin_scale1_klf4,
                                compressor=compress_jpegxr85)
        neuro_cfg9 = NeuroCodec(as_=as_,
                                vae=kl_f16,
                                quant=quant_lin_scale1_klf16,
                                compressor=compress_avif75)
        neuro_cfg10 = NeuroCodec(as_=as_,
                                 vae=kl_f16,
                                 quant=quant_lin_klf16,
                                 compressor=compress_avif80)
        neuro_cfg11 = NeuroCodec(as_=as_,
                                 vae=kl_f16,
                                 quant=quant_lin_klf16,
                                 compressor=compress_jpegxr70)
        neuro_cfg12 = NeuroCodec(as_=as_,
                                 vae=kl_f16,
                                 quant=quant_pow_scale1_klf16,
                                 compressor=compress_jpegxr70)
        neuro_cfg13 = NeuroCodec(as_=as_,
                                 vae=kl_f16,
                                 quant=quant_pow_scale1_klf16,
                                 compressor=compress_jpegxr75)
        neuro_cfg14 = NeuroCodec(as_=as_,
                                 vae=kl_f16,
                                 quant=quant_pow_scale1_klf16,
                                 compressor=compress_jpegxr80)

        self._configurations = {
            1: neuro_cfg1,
            2: neuro_cfg2,
            3: neuro_cfg3,
            4: neuro_cfg4,
            5: neuro_cfg5,
            6: neuro_cfg6,
            7: neuro_cfg7,
            8: neuro_cfg8,
            9: neuro_cfg9,
            10: neuro_cfg10,
            11: neuro_cfg11,
            12: neuro_cfg12,
            13: neuro_cfg13,
            14: neuro_cfg14,
        }

    def get_configuration(self, cfg_num):
        """Получить нейросетевой кодек конфигурации."""

        cfg_data = self._configurations.get(cfg_num, None)

        return cfg_data
