from dataset import LowLightDataset
import matplotlib
from torch import nn
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import logging

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=log_level)


def log_base(base, x):
    return torch.log(x)/torch.log(torch.tensor(float(base)))

dataset = LowLightDataset()

class LogTransformationConv(nn.Module):
    
    def __init__(self):
        super(LogTransformationConv, self).__init__()
        ## scales are mentioned in the paper. Doing it exactly as paper says.
        ## Sometime, I'm not but here I'm.
        self.log_transformation_scales = [1,10,100,300]
        self.conv_fuse = nn.Conv2d(3*len(self.log_transformation_scales), 3, kernel_size=1)
        self.conv_out = nn.Conv2d(3, 3, kernel_size=3)
    
    def forward(self, input):
        
        log_transformed = []
        for log_transformation_scale in self.log_transformation_scales:
            log_transformed.append(log_base(log_transformation_scale+1, log_transformation_scale+1 * input))
            
        concated_log_transformation = torch.cat(log_transformed, dim=1)
        logging.debug(f'concated log transformation shape: {concated_log_transformation.size()}',)
        fused = self.conv_fuse(concated_log_transformation)
        logging.debug(f'fused log transformation shape: {fused.size()}')
        output = self.conv_out(fused)
        logging.debug(f'log transformation layer output: {output.size()}')
        return output
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.log_transformation_layer = LogTransformationConv()
        
    def forward(self, input):
        log_transformed = self.log_transformation_layer(input)    
    
    
def entry_point(batch_size = 64):
    dataset = LowLightDataset()
    
    train_size = len(dataset) * 0.8
    test_size = len(dataset) - train_size
    
    train_data, test_data = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size, drop_last=True)

 
        
