import os
import sys
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from dependence.util import instantiate_from_config


mnames = ("kl-f4", "kl-f16")
ts_base = "dependence/ts/"
onnx_base = "dependence/onnx/"
nominal_type = torch.float16
# moutput_dict = {
#     "kl-f4": torch.randn(1, 3, 128, 128, dtype=nominal_type, device='cuda'),
#     "kl-f16": torch.randn(1, 16, 32, 32, dtype=nominal_type, device='cuda'),
# }


def wrap_encode(encode_func):
    def wrapper(*args, **kwargs):
        gauss = encode_func(*args, **kwargs)
        latent = gauss.sample().type(torch.float16)
        return latent

    return wrapper


def create_model(config: str, ckpt: str, nominal_type):
    """Создание оптимизированной модели.
    config -- файл конфигурации.
    ckpt -- файл весов."""

    config = OmegaConf.load(f"{config}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.eval()
    model = model.to(nominal_type).cuda()

    return model


def optimize(model_name):
    """kl-f4 kl-f16"""
    decoder_name = "{}_decoder.ts".format(model_name)
    encoder_name = "{}_encoder.ts".format(model_name)
    decoder_onnx_name = "{}_decoder.onnx".format(model_name)
    encoder_onnx_name = "{}_encoder.onnx".format(model_name)
    config_path = "dependence/config/{}.yaml".format(model_name)
    ckpt_path = "dependence/ckpt/{}.ckpt".format(model_name)

    decoder_path = os.path.join(ts_base, decoder_name)
    encoder_path = os.path.join(ts_base, encoder_name)
    is_decoder = os.path.isfile(decoder_path)
    is_encoder = os.path.isfile(encoder_path)
    decoder_onnx_path = os.path.join(onnx_base, decoder_onnx_name)
    encoder_onnx_path = os.path.join(onnx_base, encoder_onnx_name)
    is_onnx_decoder = os.path.isfile(decoder_onnx_path)
    is_onnx_encoder = os.path.isfile(encoder_onnx_path)

    print("Оптимизируем кодер...")

    encoder_input = torch.randn(1, 3, 512, 512, dtype=nominal_type, device='cuda')
    if is_encoder:
        encoder_model = torch.jit.load(encoder_path).cuda()
    else:
        model = create_model(config_path, ckpt_path, nominal_type)
        model.encode = wrap_encode(model.encode)  # Костыль
        model.forward = model.encode
        model._trainer = pl.Trainer()
        inp = [encoder_input]
        traced_model = torch.jit.trace(model, inp)
        torch.jit.save(traced_model, encoder_path)
        encoder_model = traced_model.cuda()
    encoder_model.eval()
    decoder_input_ = encoder_model(encoder_input)
    decoder_input = torch.randn(*decoder_input_.shape, dtype=nominal_type, device='cuda')

    print("Кодер преобразован в TorchScript.")
    print("Оптимизируем декодер...")

    if is_decoder:
        decoder_model = torch.jit.load(decoder_path).cuda()
    else:
        model = create_model(config_path, ckpt_path, nominal_type)
        model.forward = model.decode
        model._trainer = pl.Trainer()
        inp = [decoder_input]
        traced_model = torch.jit.trace(model, inp)
        torch.jit.save(traced_model, decoder_path)
        decoder_model = traced_model.cuda()
    decoder_model.eval()

    print("Декодер преобразован в TorchScript.")
    print("Преобразуем кодировщик в ONNX...")

    # onnx_encoder = torch.onnx.dynamo_export(encoder_model, encoder_input)
    # onnx_encoder.save(encoder_onnx_path)
    # if not is_onnx_encoder:
    torch.onnx.export(encoder_model, encoder_input, encoder_onnx_path,
                      export_params=True, do_constant_folding=True,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    print("Кодировщик преобразован в ONNX.")
    print("Преобразуем декодировщик в ONNX...")

    # onnx_decoder = torch.onnx.dynamo_export(decoder_model, decoder_input)
    # onnx_decoder.save(decoder_onnx_path)
    # if not is_onnx_decoder:
    torch.onnx.export(decoder_model, decoder_input, decoder_onnx_path,
                      export_params=True, do_constant_folding=True,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    print("Декодировщик преобразован в ONNX.")


for mname in mnames:
    print("\n\n=== Оптимизируем {} ===\n".format(mname))

    optimize(mname)
