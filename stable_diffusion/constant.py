import platform

from constants.constant import DEVICE

PRETRAINED_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-2-1"
<<<<<<< Updated upstream
TORCH_DEVICE = DEVICE
HUGGINGFACE_TOKEN = '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'
MAXSTAPEDENOISE = 1
=======
TORCH_DEVICE = "mps" if (platform.system() == "Darwin") else "cuda"
HUGGINGFACE_TOKEN = ''
MAXSTAPEDENOISE = 2
>>>>>>> Stashed changes
