import os
import torch


def data_denorm(data, avg, std):
    std = std.type(torch.cuda.FloatTensor)
    avg = avg.type(torch.cuda.FloatTensor)

    # if std == 0, change to 1.0 for nothing happen
    std = torch.where(std == torch.tensor(0, dtype=torch.float32).cuda(),
                      torch.tensor(1, dtype=torch.float32).cuda(),
                      std)

    # change the size of std and avg
    std = torch.permute(std.repeat(data.shape[1], data.shape[2], 1), [2, 0, 1])
    avg = torch.permute(avg.repeat(data.shape[1], data.shape[2], 1), [2, 0, 1])

    data = torch.mul(data, std) + avg

    return data


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation) / 2)



