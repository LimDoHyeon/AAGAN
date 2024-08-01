import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def time_warp(costs):
    dtw = np.zeros_like(costs)
    dtw[0, 1:] = np.inf
    dtw[1:, 0] = np.inf
    eps = 1e-4
    for i in range(1, costs.shape[0]):
        for j in range(1, costs.shape[1]):
            dtw[i, j] = costs[i, j] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw


def align_from_distances(distance_matrix, debug=False):
    # for each position in spectrum 1, returns best match position in spectrum2
    # using monotonic alignment
    dtw = time_warp(distance_matrix)

    i = distance_matrix.shape[0] - 1
    j = distance_matrix.shape[1] - 1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min([(i - 1, j), (i, j - 1), (i - 1, j - 1)], key=lambda x: dtw[x[0], x[1]])

    if debug:
        visual = np.zeros_like(dtw)
        visual[range(len(results)), results] = 1
        plt.matshow(visual)
        plt.show()

    return results


def DTW_align(input_data, target_data):
    for j in range(len(input_data)):
        dists = torch.cdist(torch.transpose(input_data[j], 1, 0), torch.transpose(target_data[j], 1, 0))
        alignment = align_from_distances(dists.T.cpu().detach().numpy())
        input_data[j, :, :] = input_data[j, :, alignment]

    return input_data


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


def save_checkpoint(state, best_point, save_path, filename):
    """
    Save model checkpoint.
    :param state: Model state
    :param best_point: The best checkpoint
    :param save_path: The path for saving
    :param filename: The filename for saving
    """
    save_path = str(save_path)
    filename = str(filename)

    torch.save(state, os.path.join(save_path, filename))
    # If this checkpoint is the best so far, store a copy so that it doesn't get overwritten by a worse checkpoint
    if best_point:
        torch.save(state, os.path.join(save_path, 'BEST_' + filename))
