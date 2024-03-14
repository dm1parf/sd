pip install gdown

cd model/
mkdir pretrained_models
cd pretrained_models

# DMVFN
# city dataset
gdown 1jILbS8Gm4E5Xx4tDCPZh_7rId0eo8r9W
# kitti dataset
gdown 1WrV30prRiS4hWOQBnVPUxdaTlp9XxmVK
# vimeo dataset
gdown 14_xQ3Yl3mO89hr28hbcQW3h63lLrcYY0

# DMVFN_optim
gdown 1LaUkYOyKARHhu6vSxBn-cdozgwbywODP

tar -xf dmvfn_optimised_512_fp16.tar

rm dmvfn_optimised_512_fp16.tar

# IFNet
gdown 1dLPJRe3l3uDniihosbqi-940biPEktFN

# RAFT
gdown 1gBpZIYOZaK1D9rANDuTos6jsn3uLwQoL

cd ../..