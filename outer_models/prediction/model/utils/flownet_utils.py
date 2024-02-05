import numpy as np

def fwd2bwd(fwd_flow):
    # print(fwd_flow, '\n\n', fwd_flow.dtype)
    _, _, h, w = fwd_flow.shape

    fwd_flow = fwd_flow.astype(int)

    bwd_flow = np.zeros((1, 2, h, w))
    flags = np.zeros((1,2,h,w))

    i, j = np.arange(h), np.arange(w)

    shift_x = fwd_flow[0, 0]
    shift_y = fwd_flow[0, 1]

    target_x = j + shift_x
    # target_y = np.round(i + shift_y)
    target_y = (i + shift_y.T).T
    # target_y = shift_y

    # boundary checking
    target_x = np.maximum(np.minimum(target_x, w - 1), 0)
    target_y = np.maximum(np.minimum(target_y, h - 1), 0)
    
    # bwd_flow[0, 0, target_y, target_x] -= shift_x
    # bwd_flow[0, 1, target_y, target_x] -= shift_y

    np.add.at(bwd_flow, (0, 0, target_y, target_x), -shift_x)
    np.add.at(bwd_flow, (0, 1, target_y, target_x), -shift_y)

    # flags[0, 0, target_y, target_x] += 1
    # flags[0, 1, target_y, target_x] += 1

    np.add.at(flags, (0, 0, target_y, target_x), 1)
    np.add.at(flags, (0, 1, target_y, target_x), 1)

    flags[flags < 1] = 1

    bwd_flow /= flags

    return bwd_flow