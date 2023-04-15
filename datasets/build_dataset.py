import logging
import numpy as np

# from .SIIM import *
# from .CIFAR10Dataset import *
from .SAT import * 

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


# cifar10_tsfms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

def build_cvae_dataset(dataset_name, data_path, cvae_batch_size):
    logger = logging.getLogger()
    logger.info("Build CVAE dataset for {}".format(dataset_name))
    
    if dataset_name == "LEVIRCDPlus":
        
        train_normals, val_normals, test_normals, val_anomalies, test_anomalies = get_satellite_data()
        
        train_set = Satellite_Dataset(train_normals, split="train")
        validate_set = Satellite_Dataset(val_normals, split="val")
        test_set = Satellite_Dataset(test_normals + test_anomalies, split="test")
        
    cvae_dataloaders = {'train': DataLoader(train_set, batch_size = cvae_batch_size, shuffle = True),
                      'val': DataLoader(validate_set, batch_size = cvae_batch_size),
                      'test': DataLoader(test_set, batch_size = cvae_batch_size)}
    cvae_dataset_sizes = {'train': len(train_set), 'val': len(validate_set), 'test':len(test_set)}
        
    return cvae_dataloaders, cvae_dataset_sizes
