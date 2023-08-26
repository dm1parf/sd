import torch


def get_grid(batchsize, rows, cols, gpu_id=0, dtype=torch.float32):
    hor = torch.linspace(-1.0, 1.0, cols)
    hor.requires_grad = False
    hor = hor.view(1, 1, 1, cols)
    hor = hor.expand(batchsize, 1, rows, cols)
    ver = torch.linspace(-1.0, 1.0, rows)
    ver.requires_grad = False
    ver = ver.view(1, 1, rows, 1)
    ver = ver.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([hor, ver], 1)
    t_grid.requires_grad = False

    if dtype == torch.float16: t_grid = t_grid.half()
    return t_grid.cuda(gpu_id)

def grid_sample(input1, input2, mode='bilinear', align_corners=True, resample_method = 'border'):
    return torch.nn.functional.grid_sample(input1, input2, mode=mode, padding_mode=resample_method, align_corners=align_corners)

def resample(image, flow, mode='bilinear'):        
    b, c, h, w = image.size()        
    grid = get_grid(b, h, w, gpu_id=flow.get_device(), dtype=flow.dtype)            
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)        
    #print(flow.size())
    final_grid = (grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
    #print("final_grid", final_grid.size())
    output = grid_sample(image, final_grid, mode, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(image.size())).cuda()
    mask = grid_sample(mask, final_grid, align_corners=True)
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    return output*mask


# normalize flow + warp = resample
def warp(im, flow, padding_mode='border'):
    '''
    requires absolute flow, normalized to [-1, 1]
        (see `normalize_flow` function)
    '''
    warped = torch.nn.functional.grid_sample(im, flow, padding_mode=padding_mode, align_corners=True)

    return warped