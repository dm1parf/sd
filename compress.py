import gzip
import lzma
import zstandard
import pyhuffman
import model_factory
from common.logging_sd import configure_logger
from constants.constant import Models, lossless_compression_alg

sd = model_factory.new_model(Models.SD.value)
unet = model_factory.new_model(Models.UNET.value)
encoder = model_factory.new_model(Models.ENCODER.value)
logger = configure_logger(__name__)


def run_coder(img):
    bin_quantized = sd.quantize_img(img)
    if lossless_compression_alg == 'lzma':
        return lzma.compress(bin_quantized)
    elif lossless_compression_alg == 'gzip':
        return gzip.compress(bin_quantized)
    elif lossless_compression_alg == 'zstd':
        return zstandard.compress(bin_quantized)
    elif lossless_compression_alg == 'huffman':
        return pyhuffman.encode(bin_quantized)
    else:
        return bin_quantized


def run_decoder(img):
    if lossless_compression_alg == 'lzma':
        bin_quantized = lzma.decompress(img)
    elif lossless_compression_alg == 'gzip':
        bin_quantized = gzip.decompress(img)
    elif lossless_compression_alg == 'zstd':
        bin_quantized = zstandard.decompress(img)
    elif lossless_compression_alg == 'huffman':
        bin_quantized = pyhuffman.decode(img)
    else:
        bin_quantized = img
    rest_img = sd.uncompress(bin_quantized)
    return rest_img
    # return unet.restoration(rest_img)
