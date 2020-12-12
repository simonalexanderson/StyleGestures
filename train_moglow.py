"""Train script.

Usage:
    train_moglow.py <hparams> <dataset>
"""
import os
import motion
import numpy as np
import datetime

from docopt import docopt
from torch.utils.data import DataLoader, Dataset
from glow.builder import build
from glow.trainer import Trainer
from glow.generator import Generator
from glow.config import JsonConfig
from torch.utils.data import DataLoader

if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset = args["<dataset>"]
    assert dataset in motion.Datasets, (
        "`{}` is not supported, use `{}`".format(dataset, motion.Datasets.keys()))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    hparams = JsonConfig(hparams)
    dataset = motion.Datasets[dataset]
    
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                 .replace(":", "")\
                                 .replace(" ", "_")
    log_dir = os.path.join(hparams.Dir.log_root, "log_" + date)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
		
    print("log_dir:" + str(log_dir))
    
    is_training = hparams.Infer.pre_trained == ""
    
    data = dataset(hparams, is_training)
    x_channels, cond_channels = data.n_channels()

    # build graph
    built = build(x_channels, cond_channels, hparams, is_training)
            
    if is_training:
        # build trainer
        trainer = Trainer(**built, data=data, log_dir=log_dir, hparams=hparams)
        
        # train model
        trainer.train()
    else:
        # Synthesize a lot of data. 
        generator = Generator(data, built['data_device'], log_dir, hparams)
        if "temperature" in hparams.Infer:
            temp = hparams.Infer.temperature
        else:
            temp = 1
            
        # We generate x times to get some different variations for each input
        for i in range(5):            
            generator.generate_sample(built['graph'],eps_std=temp, counter=i)
            

