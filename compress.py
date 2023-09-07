import gzip
import lzma
import zlib
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
    elif lossless_compression_alg == 'deflate':
        compress_obj = zlib.compressobj(level=6, method=zlib.DEFLATED)
        compress_data = compress_obj.compress(bin_quantized)
        compress_data += compress_obj.flush()
        return compress_data
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
    elif lossless_compression_alg == 'deflate':
        decompress = zlib.decompressobj()
        bin_quantized = decompress.decompress(img)
        bin_quantized += decompress.flush()
    else:
        bin_quantized = img
    rest_img = sd.uncompress(bin_quantized)
    return rest_img
    # return unet.restoration(rest_img)
