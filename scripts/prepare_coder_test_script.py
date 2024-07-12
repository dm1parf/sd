import os
import cv2
import torch
import time
import sys
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
from utils.config import ConfigManager

###ПРУНИНГ#########
import torch.nn as nn
import torch.nn.utils.prune as prune
###################

config_path = os.path.join("scripts", "encoder_config.ini")
config_mng = ConfigManager(config_path)
as_ = config_mng.get_as_worker()
vae = config_mng.get_autoencoder_worker()


def main():
    global config_mng

    encoder_path = "dependence/ts/pruned_encoder.ts"
    encoder = torch.jit.load(encoder_path).cuda()
    encoder.eval()

    # ЗДЕСЬ КАКОЙ-НИБУДЬ ПРУНИНГ

    def is_conv2d(module):
      return isinstance(module, nn.Conv2d) or (
          isinstance(module, torch.jit.RecursiveScriptModule) and
          module.original_name == 'Conv2d'
      )

    def prune_conv2d(module, amount):
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
        elif (isinstance(module, torch.jit.RecursiveScriptModule) 
                and module.original_name == 'Conv2d'):
            weight = module.weight
            mask = torch.ones_like(weight)
            n = int(amount * weight.nelement())
            _, indices = torch.topk(
                            torch.abs(weight).view(-1), k=n, largest=False)
            mask.view(-1)[indices] = 0
            pruned_weight = weight * mask
            with torch.no_grad():
                module.weight.copy_(pruned_weight)

    def prune_model(model, amount):
        for name, module in model.named_modules():
            if is_conv2d(module):
                # print(f"Pruning Conv2d layer: {name}")
                prune_conv2d(module, amount)

    pruning_enable = False
    prune_amount = 0.3
    if pruning_enable:
        prune_model(encoder, amount=prune_amount) # ИЗМЕНЯЕТСЯ САМ encoder

    #######################################################################


    input_image = "test.jpg"
    frame = cv2.imread(input_image)

    torch.cuda.synchronize()
    a = time.time()
    frame, _ = as_.prepare_work(frame)

    latent_img = encoder.forward(frame)
    # torch.jit.save(encoder, "pruned_encoder.ts")

    torch.cuda.synchronize()
    b = time.time()
    all_time = b - a
    coder_fps = 1 / all_time

    print(all_time)
    print(coder_fps)

    """
    input_video = "materials/dataset/air/2/7.mp4"
    cap = cv2.VideoCapture(input_video)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(input_video)
            ret, frame = cap.read()
        a = time.time()
        latent_img = encoder_pipeline(frame)
        b = time.time()
        all_time = b - a
        coder_fps = 1 / all_time
    """


if __name__ == "__main__":
    main()
