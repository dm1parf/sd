import torch
from torch.autograd import Variable

import cv2
import numpy as np

import os
import argparse
import traceback

from pathlib import Path
from copy import deepcopy

from math import ceil

from tqdm.auto import tqdm

from .DMVFN import DMVFN as DMVFN_model

from .VPvI import IFNet, resample, regionfill, Consistency

from .RAFT import RAFT

from .VPTR import VPTREnc, VPTRDec, VPTRFormerFAR

from .utils import convertModuleNames
from .utils import fwd2bwd


### TODO add full support for cpu


class VPTR: # Video Prediction TransformeR
    """
    No pretrained models yet
    Need to pass height and width
    """

    def __init__(self,
                 load_path_Enc=None,
                 load_path_Dec=None,
                 load_path_Transformer=None,
                 h = 720,
                 w = 1280,
                 num_past_frames = 2,
                 num_future_frames = 2,
                 n_downsampling = 6,
                 device="cuda"):
        
        self.device = device
        self.h = h
        self.w = w
        
        # resume_ckpt = Path('checkpoints/BAIR_FAR.tar') #The trained Transformer checkpoint file
        # resume_AE_ckpt = Path('checkpoints/BAIR_AE.tar') #The trained AutoEncoder checkpoint file

        encH, encW, encC = (ceil(h / (2**n_downsampling)), 
                            ceil(w / (2**n_downsampling)), 
                            528)
        img_channels = 3
        # N = 2
        rpe = True

        self.VPTR_Enc = VPTREnc(img_channels, feat_dim = encC, n_downsampling = n_downsampling, padding_type = 'zero').to(device)

        self.VPTR_Dec = VPTRDec(img_channels, feat_dim = encC, n_downsampling = n_downsampling, out_layer = 'Sigmoid', padding_type = 'zero').to(device) 

        self.VPTR_Transformer = VPTRFormerFAR(num_past_frames, num_future_frames, encH=encH, encW = encW, d_model=encC, 
                                              nhead=8, num_encoder_layers=12, dropout=0.1, 
                                              window_size=4, Spatial_FFN_hidden_ratio=4, rpe=rpe
                                              ).to(device)

        self.VPTR_Enc = self.VPTR_Enc.eval()
        self.VPTR_Dec = self.VPTR_Dec.eval()
        self.VPTR_Transformer = self.VPTR_Transformer.eval()

        if (load_path_Enc is not None) or (load_path_Dec is not None) or (load_path_Transformer is not None):
            raise NotImplementedError("Loading models not implemented yet")


        # loss_name_list = ['T_MSE', 'T_GDL', 'T_gan', 'T_total', 'Dtotal', 'Dfake', 'Dreal']

        # loss_dict, start_epoch = resume_training({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec}, {}, resume_AE_ckpt, loss_name_list)
        # if resume_ckpt is not None:
        #     loss_dict, start_epoch = resume_training({'VPTR_Transformer': VPTR_Transformer}, 
        #                                             {}, resume_ckpt, loss_name_list)



    @torch.no_grad()
    def evaluate(self, imgs : list[np.ndarray]):
        for i in range(len(imgs)):
            # HWC (height width channel) -> CHW (channel height width)
            imgs[i] = cv2.resize(imgs[i], (self.w, self.h))
            imgs[i] = (imgs[i].transpose(2, 0, 1).astype('float32'))
        
        img = torch.tensor(np.array(imgs))
        img = img.unsqueeze(0) # NCHW
        img = img.to(self.device) / 255.

        # print(img.shape)

        past_gt_feats = self.VPTR_Enc(img)
        pred_feats = self.VPTR_Transformer(past_gt_feats)
        pred = self.VPTR_Dec(pred_feats[:, -1:, ...])
    
        pred = np.array(pred.cpu().squeeze() * 255).transpose(1, 2, 0) # CHW -> HWC
        pred = pred.astype("uint8")
        
        return pred


class DMVFN: # Dynamic Multi-Scale Voxel Flow Network
    """
    to install pretrained models:
        pip install gdown

        # city dataset
        gdown 1jILbS8Gm4E5Xx4tDCPZh_7rId0eo8r9W
        
        # kitti dataset
        gdown 1WrV30prRiS4hWOQBnVPUxdaTlp9XxmVK
        
        # vimeo dataset
        gdown 14_xQ3Yl3mO89hr28hbcQW3h63lLrcYY0

    """
    def __init__(self, load_path, device="cuda"):
        self.device = device
        self.dmvfn = DMVFN_model()
        self.dmvfn.to(self.device)

        # load model
        state_dict = torch.load(load_path)
        model_state_dict = self.dmvfn.state_dict()
        
        for k in model_state_dict.keys():
            model_state_dict[k] = state_dict['module.' + k]

        self.dmvfn.load_state_dict(model_state_dict)

    @torch.no_grad()
    def evaluate(self, imgs : list[np.ndarray], scale_list : list[int] = [4,4,4,2,2,2,1,1,1]):
        
        for i in range(len(imgs)):
            # HWC (height width channel) -> CHW (channel height width)
            imgs[i] = torch.tensor(imgs[i].transpose(2, 0, 1).astype('float32'))
        
        img = torch.cat(imgs, dim=0)
        img = img.unsqueeze(0) # .unsqueeze(0) # NCHW
        img = img.to(self.device, non_blocking=True) / 255.

        try:
            pred = self.dmvfn(img, scale=scale_list, training=False) # 1CHW
        except RuntimeError:
            ### TODO Нужно точно определить какое разрешение принимает модель
            print(traceback.format_exc())
            raise RuntimeError("Image resolution is not compatible")

        # если модель ничего не возвращает, то возращаем последний фрейм
        if len(pred) == 0:
            pred = imgs[:, 0]
        else:
            pred = pred[-1]
        
        pred = np.array(pred.cpu().squeeze() * 255).transpose(1, 2, 0) # CHW -> HWC
        pred = pred.astype("uint8")
        
        return pred


class VPvI: # Video Prediction via Interpolation
    """
    to install pretrained models:
        pip install gdown

        # IFNet
        gdown 1dLPJRe3l3uDniihosbqi-940biPEktFN

        # RAFT
        gdown 1gBpZIYOZaK1D9rANDuTos6jsn3uLwQoL
    """

    def __init__(self,
                 model_load_path = "RAFT/pretrained_models/flownet.pkl",
                 # model_load_path = "train_log/flownet.pkl",
                 flownet_load_path = "RAFT/pretrained_models/raft-kitti.pth",
                 scale_list : list[int] = [8, 4, 2, 1],
                 iters : int = 1,
                 consist_weight = 0.1,
                 interp_weight = 1,
                 lr = 0.0001,
                 # use_consistency_loss = False,
                 fast_approximation = False,
                 define_random_flow = False,
                 device="cuda",):

        self.device = device
        self.interp_model = IFNet()
        self.interp_model.to(device)

        for param in self.interp_model.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            self.interp_model.load_state_dict(convertModuleNames(torch.load(model_load_path)))
        else:
            self.interp_model.load_state_dict(convertModuleNames(torch.load(model_load_path, map_location ='cpu')))

        # Emulate arguments for RAFT
        args = argparse.Namespace()
        args.model = Path(flownet_load_path)
        args.small = False
        args.mixed_precision = True
        args.alternate_corr = False

        self.flowNet = torch.nn.DataParallel(RAFT(args))
        self.flowNet.to(device)
        self.flowNet.load_state_dict(torch.load(flownet_load_path, map_location ='cpu'))

        self.flowNet = self.flowNet.module
        for param in self.flowNet.parameters():
            param.requires_grad = False

        self.consist_weight = consist_weight
        self.interp_weight = interp_weight
        self.lr = lr
        self.fast_approximation = fast_approximation
        # self.use_consistency_loss = use_consistency_loss
        self.define_random_flow = define_random_flow

        self.iters = iters
        self.scale_list = scale_list


        self.l1_loss = torch.nn.L1Loss()
        self.flow_consistency_loss = Consistency()
        # self.smoothness_loss = SmoothLoss()
        # self.tv_loss = TVLoss()


    def evaluate(
        self, imgs : list[np.ndarray]
        ) -> np.ndarray:

        assert len(imgs) == 2, f"only 2 images can be accepted as input, {len(imgs)} were passed"

        # HWC -> CHW
        im1 = torch.tensor(imgs[0].transpose(2, 0, 1)).unsqueeze(0).cuda().float() / 255.
        im2 = torch.tensor(imgs[1].transpose(2, 0, 1)).unsqueeze(0).cuda().float() / 255.

        if self.define_random_flow:
            flow_2to1 = (0.1**0.5)*torch.randn((1, 2, *im1.shape[-2:]))

        else:
            _, flow_2to1 = self.flowNet(im2 * 255, im1 * 255, iters=10, test_mode=True)

        if self.fast_approximation:
            bwd_flow_3to2 = - flow_2to1
        
        else:
            flow_2to3 = - flow_2to1.detach().cpu().numpy()

            bwd_flow_3to2 = fwd2bwd(flow_2to3)

            bwd_flow_3to2 = torch.from_numpy(bwd_flow_3to2)

        if not self.iters: # don't do any optimisation
            pred = resample(im2, bwd_flow_3to2.cuda().float())
            pred = np.array(pred.detach().cpu().squeeze() * 255).transpose(1, 2, 0) # CHW -> HWC
            pred = pred.astype("uint8")

            return pred

        flow = Variable(bwd_flow_3to2.cuda().float(), requires_grad=True)
        optimizer = torch.optim.Adam(
            [{'params': flow, 'lr': self.lr}]
        )

        # frame_pred = resample(im2, flow)

        for it in range(self.iters):
            optimizer.zero_grad()

            frame_pred = resample(im2, flow)

            imgs = torch.cat((im1, frame_pred), 1)

            # interp_result, flow_2to3_pred, warp_pred = self.interp_model(imgs, scale_list)
            flow_2to3_pred, mask, merged = self.interp_model(
                imgs, 0.5, self.scale_list)

            flow_2to3_pred = flow_2to3_pred[3][:, 2:4]
            
            interp_result = merged[3]
            
            # Loss functions
            loss = 0.0

            loss_interp = self.l1_loss(interp_result, im2) * self.interp_weight
            loss += loss_interp
            # loss = loss_interp

            # if self.use_consistency_loss:
            fwd_mask, bwd_mask, consist_loss = self.flow_consistency_loss(flow_2to3_pred, flow, 1)
            consist_loss = consist_loss * self.consist_weight
            loss += consist_loss

            loss.backward()
            optimizer.step()

        frame_pred_before = resample(im2.detach().float(), flow.float())

        im2_vis = im2.detach().cpu().numpy()
        flow_ = flow.detach().cpu().numpy()
        
        bwd_mask_ = 1 - bwd_mask.detach().cpu().numpy().astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        bwd_mask_x = cv2.dilate(bwd_mask_[0,0,...], kernel, iterations=1)
        bwd_mask_y = cv2.dilate(bwd_mask_[0,1,...], kernel, iterations=1)
        flow_x = regionfill(flow_[0,0,...], bwd_mask_x.astype(int))
        flow_y = regionfill(flow_[0,1,...], bwd_mask_y.astype(int))
        flow_new = np.concatenate([flow_x[None,None,...], flow_y[None,None,...]], axis=1)
        flow_new = torch.from_numpy(flow_new).cuda()

        frame_pred = resample(im2.detach().float(), flow_new.float())

        pred = frame_pred

        pred = np.array(pred.detach().cpu().squeeze() * 255).transpose(1, 2, 0) # CHW -> HWC
        pred = pred.astype("uint8")

        return pred



class Model:
    """
    model = Model(DMVFN(load_path = "./pretrained_models/dmvfn_city.pkl"))
    
    model = Model(
                VPvI(model_load_path = "./pretrained_models/flownet.pkl",
                     flownet_load_path = "./pretrained_models/raft-kitti.pth"))
    """

    def __init__(self, model):
        self.model = model
        self.device = self.model.device


    def predict(self, imgs : list[np.ndarray], num_frames_to_predict : int = 1) -> list[np.ndarray]:
        """
        args:
            imgs : list[np.ndarray] - фреймы, по которым будет происходить предсказание
            num_frames_to_predict : int - количество кадров, которое нужно предсказать
        
        returns:
            img : np.ndarray - Предсказанный фрейм если num_frames_to_predict = 1
            imgs : np.ndarray | list[np.ndarray] - Предсказанные фреймы или один фрейм, если num_frames_to_predict = 1

        examples:
            >>> img1, img2 = cv2.imread("1.jpg"), cv2.imread("2.jpg")
            >>> img1, img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            >>> img1, img2 = cv2.resize(img1, (1024, 512)), cv2.resize(img2, (1024, 512))
            >>> pred = model.predict([img1, img2])
            >>> pred.shape
            (512, 1024, 3)

            >>> pred = model.predict([img1, img2], 10)
            >>> len(pred)
            10
        """

        imgs = list(deepcopy(imgs)) ### do not modify the input list

        if num_frames_to_predict == 1:
            return self.model.evaluate(imgs)

        # how many frames are passed to model
        frames_to_model = len(imgs)

        for i in range(num_frames_to_predict):
            img_pred = self.model.evaluate(imgs[-frames_to_model:])
            imgs.append(img_pred)

        # do not return input frames
        return imgs[frames_to_model:]


    def predictVideo(self,
                     video_path : str,
                     save_path : str,
                     real_fake_pattern : list[str] = [],
                     real : int = 1,
                     fake : int = 1,
                     frames_to_model : int = 2,
                     w : int = 1024,
                     h : int = 1792,):
        """
        args:
            video_path : str - Путь до видео, фреймы в котором надо предсказать
            save_path : str or None - Куда сохранить новое видео, если None, то видео сохраняться не будет
            real_fake_pattern : list[str] - Паттерн, который указывает какие кадры должны быть предсказаны, паттерн должен желательно начинаться с "fake"
            real : int - Количество кадров в паттерне, которые считываются с реалього видео, не учитывается при передаче real_fake_pattern
            fake : int - Количество кадров в паттерне, которые будут предсказаны, не учитывается при передаче real_fake_pattern
            frames_to_model : int - Сколько кадров передаётся в модель для предсказания
            w : int - Ширина видео
            h : int - Высота видео

        returns:
            frames : list[np.array] - все кадры видео
            real_fake_mask : list[str] - маска для frames, где реальные кадры отмечены "real", предсказанные кадры "fake"

        examples:
            >>> frames, mask = model.predictVideo("path/to/video", "where/to/save", real = 2, fake = 1, frames_to_model = 2)
            >>> mask
            ["real", "real", "fake", "real", "real", "fake", "real", "real", "fake", ...]

            >>> frames, mask = model.predictVideo("path/to/video", "where/to/save", real = 1, fake = 1, frames_to_model = 2)
            >>> mask
            ["real", "real", "fake", "real",  "fake", "real", "fake", ...]

            >>> frames, mask = model.predictVideo("path/to/video", "where/to/save", real = 2, fake = 2, frames_to_model = 4)
            >>> mask
            ["real", "real", "real", "real", "fake", "fake", "real", "real", "fake", "fake", "real", "real", ...]

            >>> frames, mask = model.predictVideo("path/to/video", "where/to/save", real_fake_pattern = ["fake", "real", "real"] frames_to_model = 2)
            >>> mask
            ["real", "real", "fake", "real", "real", "fake", "real", "real", "fake", ...]

            >>> # не стоит начинать паттерн с "real", так как первоначально уже считываеются реальные кадры, а потом начинается паттерн
            >>> frames, mask = model.predictVideo("path/to/video", "where/to/save", real_fake_pattern = ["real", "fake"] frames_to_model = 2)
            >>> mask
            ["real", "real", "real", "fake", "real", "fake", "real", "fake", ...]
        """

        if not real_fake_pattern:
            real_fake_pattern = ["fake"] * fake + ["real"] * real
        
        pattern_len = len(real_fake_pattern)

        if save_path: save_path = str(save_path)
        video_path = str(video_path)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if save_path:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (h, w))

        frames : list[np.ndarray] = []
        real_fake_mask : list[str] = []

        for i in range(frames_to_model):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (h, w))
                frames.append(frame)
                real_fake_mask.append("real")
                if save_path: writer.write(frame)
            else: # failed to read video
                cap.release()
                if save_path: writer.release()
                raise RuntimeError(f"Unable to read video {video_path = }")

        try:
            for i in tqdm(range(frame_count-frames_to_model)):
                ret, frame = cap.read()

                if ret:
                    frame = cv2.resize(frame, (h, w))
                    
                    if real_fake_pattern[i % pattern_len] == "real": # read next frame
                        frames.append(frame)
                        real_fake_mask.append("real")
                        if save_path: writer.write(frame)
                        
                    else: # predict next frame
                        img_pred = self.predict(frames[-frames_to_model:])
                        frames.append(img_pred)
                        real_fake_mask.append("fake")
                        if save_path: writer.write(img_pred)

                else:
                    break
        
        except Exception:
            print(traceback.format_exc())

        # Всегда закрываем файлы, даже в случае непредвиденных ошибок, например CUDA out of memory
        finally:
            cap.release()
            if save_path: writer.release()

        return frames, real_fake_mask
