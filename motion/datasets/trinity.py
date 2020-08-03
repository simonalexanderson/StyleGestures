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

def fit_and_standardize(data):

    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler

def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled

def inv_standardize(data, scaler):      
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled        
                

class Trinity():

    def __init__(self, hparams):
        data_root = hparams.Dir.data_root

        #load data
        train_input = np.load(os.path.join(data_root, 'train_input_'+str(hparams.Data.framerate)+'fps.npz'))['clips'].astype(np.float32)
        train_output = np.load(os.path.join(data_root, 'train_output_'+str(hparams.Data.framerate)+'fps.npz'))['clips'].astype(np.float32)
        val_input = np.load(os.path.join(data_root, 'val_input_'+str(hparams.Data.framerate)+'fps.npz'))['clips'].astype(np.float32)
        val_output = np.load(os.path.join(data_root, 'val_output_'+str(hparams.Data.framerate)+'fps.npz'))['clips'].astype(np.float32)

        #use this to generate visualizations for network tuning. It contains the same data as val_input, but sliced into longer 20-sec exerpts
        test_input = np.load(os.path.join(data_root, 'dev_input_'+str(hparams.Data.framerate)+'fps.npz'))['clips'].astype(np.float32)
        
        #load pipeline for convertion from motion features to BVH.
        self.data_pipe = jl.load(os.path.join(data_root, 'data_pipe_'+str(hparams.Data.framerate)+'fps.sav'))
        
        #use this to generate test data for evaluation
        #test_input = np.load(os.path.join(data_root, 'test_input_'+str(hparams.Data.framerate)+'fps.npz'))['clips'].astype(np.float32)
                       
        # make sure the test data is at least one batch size
        self.n_test = test_input.shape[0]
        n_tiles = 1+hparams.Train.batch_size//self.n_test
        test_input = np.tile(test_input.copy(), (n_tiles,1,1))

        # Standartize
        train_input, input_scaler = fit_and_standardize(train_input)
        train_output, output_scaler = fit_and_standardize(train_output)
        val_input = standardize(val_input, input_scaler)
        val_output = standardize(val_output, output_scaler)
        test_input = standardize(test_input, input_scaler)
        test_output = np.zeros((test_input.shape[0], test_input.shape[1], train_output.shape[2])).astype(np.float32)
                        
        # Create pytorch data sets
        self.train_dataset = MotionDataset(train_input, train_output, hparams.Data.seqlen, hparams.Data.n_lookahead, hparams.Data.dropout)    
        self.validation_dataset = MotionDataset(val_input, val_output, hparams.Data.seqlen, hparams.Data.n_lookahead, hparams.Data.dropout)    
        self.test_dataset = TestDataset(test_input, test_output)
        
        # Store scaler and fps
        self.scaler = output_scaler
        self.fps = hparams.Data.framerate

    def save_animation(self, control_data, motion_data, filename):
        anim_clips = inv_standardize(motion_data[:(self.n_test),:,:], self.scaler)
        np.savez(filename + ".npz", clips=anim_clips)  
        write_bvh(anim_clips, filename, )
        
    def write_bvh(self, anim_clips, filename):
        print('inverse_transform...')
        inv_data=self.data_pipe.inverse_transform(anim_clips)
        writer = BVHWriter()
        for i in range(0,anim_clips.shape[0]):
            filename_ = f'{filename}_{str(i)}.bvh'
            print('writing:' + filename_)
            with open(filename_,'w') as f:
                writer.write(inv_data[i], f, framerate=self.fps)
        
    def get_train_dataset(self):
        return self.train_dataset
        
    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset
        
