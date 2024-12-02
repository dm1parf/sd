import os
import sys
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
import torch
from torchview import draw_graph
from utils.workers import (WorkerAutoencoderKL_F16, WorkerAutoencoderKL_F4)


def visualize_model_torchview(model):
    # model = MLP()
    batch_size = 2
    # device='meta' -> no memory is consumed for visualization
    # (1, 3, 512, 512)
    # (1, 3, 128, 128)
    # (1, 16, 32, 32)
    model_graph = draw_graph(model, input_size=(1, 16, 32, 32), device='meta',
                             save_graph=True, filename="kl_f16_decoder", directory="./scripts/rare_scripts/")
    model_graph.visual_graph
    print(model_graph.visual_graph)


kl_f4 = WorkerAutoencoderKL_F4(config_path="dependence/config/kl-f4.yaml",
                               ckpt_path="dependence/ckpt/kl-f4.ckpt")
kl_f16 = WorkerAutoencoderKL_F16(config_path="dependence/config/kl-f16.yaml",
                                 ckpt_path="dependence/ckpt/kl-f16.ckpt")
klf4_model = kl_f4._model
klf4_model = klf4_model.type(torch.float32)
# klf4_model.forward = klf4_model.encode
klf4_model.forward = klf4_model.decode
klf16_model = kl_f16._model
klf16_model = klf16_model.type(torch.float32)
# klf16_model.forward = klf16_model.encode
klf16_model.forward = klf16_model.decode

# visualize_model_torchview(klf4_model)
visualize_model_torchview(klf16_model)

