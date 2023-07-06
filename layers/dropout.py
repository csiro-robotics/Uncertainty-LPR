# Code taken from: https://github.com/filipradenovic/cnnimageretrieval-pytorch
# and ported to MinkowskiEngine by Jacek Komorowski

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from typing import Union

from torch.nn.modules import Module

from MinkowskiSparseTensor import SparseTensor
from MinkowskiTensorField import TensorField


class MinkowskiDropout(Module):
    def __init__(self, dropout_rate):
        super(MinkowskiDropout, self).__init__()

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, input: Union[SparseTensor, TensorField]):

        output = self.dropout(input.F)
       
        if isinstance(input, TensorField):
            return TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )