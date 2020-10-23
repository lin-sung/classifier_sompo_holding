import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
        Base model to use image and text information to make multi-class prediction. 
    """
    def __init__(self):
        pass

    def forward(self, inputs):
        pass