import os
import sys
import numpy as np
import joblib as jl
from .motion_data import MotionDataset, TestDataset
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

module_path = os.path.abspath(os.path.join('data_processing'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pymo.writers import *

def inv_standardize(data, scaler):      
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled        
                
class Locomotion_rot():

    def __init__(self, hparams, is_training):
        data_root = hparams.Dir.data_root
        
        #load scalers.
        self.input_scaler = jl.load(os.path.join(data_root, 'input_scaler.sav'))
        self.output_scaler = jl.load(os.path.join(data_root, 'output_scaler.sav'))        
        self.data_pipe = jl.load(os.path.join(data_root, 'data_pipe_'+str(hparams.Data.framerate)+'fps.sav'))
        
        if is_training:
            #load the data. This should allready be Standartized
            train_input = np.load(os.path.join(data_root, 'train_input_'+str(hparams.Data.framerate)+'fps.npz'))['clips'].astype(np.float32)
            train_output = np.load(os.path.join(data_root, 'train_output_'+str(hparams.Data.framerate)+'fps.npz'))['clips'].astype(np.float32)
            val_input = np.load(os.path.join(data_root, 'val_input_'+str(hparams.Data.framerate)+'fps.npz'))['clips'].astype(np.float32)
            val_output = np.load(os.path.join(data_root, 'val_output_'+str(hparams.Data.framerate)+'fps.npz'))['clips'].astype(np.float32)

            # Create pytorch data sets
            self.train_dataset = MotionDataset(train_input, train_output, hparams.Data.seqlen, hparams.Data.n_lookahead, hparams.Data.dropout)    
            self.validation_dataset = MotionDataset(val_input, val_output, hparams.Data.seqlen, hparams.Data.n_lookahead, hparams.Data.dropout)    
            
            #test data for network tuning. It contains the same data as val_input, but sliced into longer 20-sec exerpts
            test_input = np.load(os.path.join(data_root, 'dev_input_'+str(hparams.Data.framerate)+'fps.npz'))['clips'].astype(np.float32)                                   
        else:
            self.train_dataset = None
            self.validation_dataset = None
            test_input = np.load(os.path.join(data_root, 'test_input_'+str(hparams.Data.framerate)+'fps.npz'))['clips'].astype(np.float32)
                       
        # make sure the test data is at least one batch size
        self.n_test = test_input.shape[0]
        n_tiles = 1+hparams.Train.batch_size//self.n_test
        test_input = np.tile(test_input.copy(), (n_tiles,1,1))

        # initialise test output with zeros (mean pose)
        self.n_x_channels = self.output_scaler.mean_.shape[0]
        self.n_cond_channels = self.n_x_channels*hparams.Data.seqlen + test_input.shape[2]*(hparams.Data.seqlen + 1 + hparams.Data.n_lookahead)
        test_output = np.zeros((test_input.shape[0], test_input.shape[1], self.n_x_channels)).astype(np.float32)
                        
        self.test_dataset = TestDataset(test_input, test_output)
        
        # Store fps
        self.fps = hparams.Data.framerate

    def save_animation(self, control_data, motion_data, filename):
        #import pdb;pdb.set_trace()
        anim_data = np.concatenate((inv_standardize(motion_data, self.output_scaler), inv_standardize(control_data, self.input_scaler)), axis=2)
        anim_clips = anim_data[:(self.n_test),:,:]
        np.savez(filename + ".npz", clips=anim_clips)  
        self.write_bvh(anim_clips, filename)
        
    def write_bvh(self, anim_clips, filename):
        print('inverse_transform...')
        inv_data=self.data_pipe.inverse_transform(anim_clips)
        writer = BVHWriter()
        for i in range(0,anim_clips.shape[0]):
            filename_ = f'{filename}_{str(i)}.bvh'
            print('writing:' + filename_)
            with open(filename_,'w') as f:
                writer.write(inv_data[i], f, framerate=self.fps)
        
    def n_channels(self):
        return self.n_x_channels, self.n_cond_channels
        
    def get_train_dataset(self):
        return self.train_dataset
        
    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset
        
