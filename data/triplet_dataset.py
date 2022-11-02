from typing import Any

import numpy as np
import torch
from torchtyping import TensorType

Tensor = torch.Tensor
Array = np.ndarray

class TripletData(torch.utils.data.Dataset):
    def __init__(self, triplets: Any, n_objects: int):
        super(TripletData, self).__init__()
        if isinstance(triplets, Tensor):
            self.triplets = triplets
        elif isinstance(triplets, Array):
            self.triplets = torch.from_numpy(triplets).type(torch.LongTensor)
        else:
            raise TypeError(f'\nData has incorrect type:{type(triplets)}\n')
        self.identity = torch.eye(n_objects).to(self.triplets.device)
    
    def encode_as_onehot(self, triplet: TensorType["k"]) -> TensorType["k", "m"]:
        """Encode a tensor of three numerical indices as a tensor of three one-hot-vectors."""
        return self.identity[triplet, :]

    def __getitem__(self, index: int) -> TensorType["k", "m"]:
        index_triplet = self.triplets[index]
        one_hot_triplet = self.encode_as_onehot(index_triplet)
        return one_hot_triplet

    def __len__(self) -> int:
        return self.triplets.shape[0]
