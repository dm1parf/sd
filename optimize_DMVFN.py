from speedster import save_model, optimize_model

from prediction.model import DMVFN

import torch

import os
import shutil


dmvfn = DMVFN("prediction/model/pretrained_models/dmvfn_city.pkl")


from tqdm.auto import tqdm

blocks_original = [dmvfn.block0, dmvfn.block1, dmvfn.block2,
                   dmvfn.block3, dmvfn.block4, dmvfn.block5,
                   dmvfn.block6, dmvfn.block7, dmvfn.block8]


blocks_optim = []


input_data = [(
        (torch.randn(1, 13, 512, 512).cuda(), torch.randn(1, 4, 512, 512).cuda()),  # input 
        # (torch.randn(1, 4, 512, 512).cuda(), torch.randn(1, 1, 512, 512).cuda()),
        )  # output
    for _ in range(200)]


for i in tqdm(range(9)):
    block_original = blocks_original[i]
    
    block_optim = optimize_model(
        block_original,
        input_data=input_data,
        optimization_time="constrained",
        metric_drop_ths=0.05,
        # dynamic_info=dynamic_info,
        ignore_compilers=["torchscript"],
    )

    blocks_optim.append(block_optim)
    

save_path = "prediction/model/pretrained_models/dmvfn_optimised_512_fp16"

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)

for i in range(len(blocks_optim)):
    save_model(blocks_optim[i], f"{save_path}/block_{i}")
