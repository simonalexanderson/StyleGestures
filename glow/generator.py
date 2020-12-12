import re
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from .models import Glow


class Generator(object):
    def __init__(self, data, data_device, log_dir, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)

        # model relative
        self.data_device = data_device
        self.seqlen = hparams.Data.seqlen
        self.n_lookahead = hparams.Data.n_lookahead
        self.data = data
        self.log_dir = log_dir

        # test batch
        self.test_data_loader = DataLoader(data.get_test_dataset(),
                                      batch_size=hparams.Train.batch_size,
                                      num_workers=1,
                                      shuffle=False,
                                      drop_last=True)
        self.test_batch = next(iter(self.test_data_loader))
        for k in self.test_batch:
            self.test_batch[k] = self.test_batch[k].to(self.data_device)
            
    def prepare_cond(self, jt_data, ctrl_data):
        nn,seqlen,n_feats = jt_data.shape
        
        jt_data = jt_data.reshape((nn, seqlen*n_feats))
        nn,seqlen,n_feats = ctrl_data.shape
        ctrl_data = ctrl_data.reshape((nn, seqlen*n_feats))
        cond = torch.from_numpy(np.expand_dims(np.concatenate((jt_data,ctrl_data),axis=1), axis=-1))
        return cond.to(self.data_device)
    
    def generate_sample(self, graph, eps_std=1.0, step=0, counter=0):
        print("generate_sample")

        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy()
        control_all = batch["control"].cpu().numpy()

        # Initialize the pose sequence with ground truth test data
        seqlen = self.seqlen
        n_lookahead = self.n_lookahead
        
        # Initialize the lstm hidden state
        if hasattr(graph, "module"):
            graph.module.init_lstm_hidden()
        else:
            graph.init_lstm_hidden()
            
        nn,n_timesteps,n_feats = autoreg_all.shape
        sampled_all = np.zeros((nn, n_timesteps-n_lookahead, n_feats))
        autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32) #initialize from a mean pose
        sampled_all[:,:seqlen,:] = autoreg
        
        # Loop through control sequence and generate new data
        for i in range(0,control_all.shape[1]-seqlen-n_lookahead):
            control = control_all[:,i:(i+seqlen+1+n_lookahead),:]
            
            # prepare conditioning for moglow (control + previous poses)
            cond = self.prepare_cond(autoreg.copy(), control.copy())

            # sample from Moglow
            sampled = graph(z=None, cond=cond, eps_std=eps_std, reverse=True)
            sampled = sampled.cpu().numpy()[:,:,0]

            # store the sampled frame
            sampled_all[:,(i+seqlen),:] = sampled
            
            # update saved pose sequence
            autoreg = np.concatenate((autoreg[:,1:,:].copy(), sampled[:,None,:]), axis=1)
            
        # store the generated animations
        self.data.save_animation(control_all[:,:(n_timesteps-n_lookahead),:], sampled_all, os.path.join(self.log_dir, f'sampled_{counter}_temp{str(int(eps_std*100))}_{str(step//1000)}k'))              
        