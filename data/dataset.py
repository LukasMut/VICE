from typing import List

import numpy as np
import torch

Tensor = torch.Tensor
Array = np.ndarray


class TripletData(torch.utils.data.Dataset):
    def __init__(self, triplets: List[List[int]], n_objects: int):
        super(TripletData, self).__init__()
        self.triplets = torch.tensor(triplets).type(torch.LongTensor)
        self.identity = torch.eye(n_objects)

    def encode_as_onehot(self, triplet: Tensor) -> Tensor:
        """Encode a triplet of indices as a matrix of three one-hot-vectors."""
        return self.identity[triplet, :]

    def __getitem__(self, index: int) -> Tensor:
        index_triplet = self.triplets[index]
        one_hot_triplet = self.encode_as_onehot(index_triplet)
        return one_hot_triplet

    def __len__(self) -> int:
        return self.triplets.shape[0]
