import numpy as np

import glob
import os
import sys
from shutil import copyfile
from audio_features import extract_melspec
import scipy.io.wavfile as wav
import joblib as jl

            
def import_and_pad(files, speech_data):
    """Imports all features and pads them to samples with equal lenth time [samples, timesteps, features]."""
                    
    max_frames = 0
    for file in files:
        print(file)
        
        # compute longest clip
        speech_data = np.load(os.path.join(speech_path, file + '.npy')).astype(np.float32)
        if speech_data.shape[0]>max_frames:
            max_frames = speech_data.shape[0]
            
        n_feats = speech_data.shape[1]
            
    out_data = np.zeros((len(files), max_frames, n_feats))
    
    fi=0
    for file in files:        
        # pad to longest clip length
        speech_data = np.load(os.path.join(speech_path, file + '.npy')).astype(np.float32)
        out_data[fi,:speech_data.shape[0], :] = speech_data
        print("No: " + str(fi) + " File: " + file)
        fi+=1
        
    return out_data
    
if __name__ == "__main__":
    '''
    Converts wav files into features and creates a test dataset.
    '''     
    # Hardcoded preprocessing params and file structure. 
    # Modify these if you want the data in some different format
    test_window_secs = 20
    window_overlap = 0.5
    fps = 20

    data_root = '../data/GENEA/source'
    audiopath = os.path.join(data_root, 'test_audio')
    processed_dir = '../data/GENEA/processed'
    test_dir = '../data/GENEA/processed/test'
    
    files = []
    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(audiopath):
        for file in sorted(f):
            if '.wav' in file:
                ff=os.path.join(r, file)
                basename = os.path.splitext(os.path.basename(ff))[0]
                files.append(basename)

    print(files)
    if len(files)==0:
        print ("no files found in: " + audiopath)
    
    speech_feat = 'melspec'
    
    # processed data will be organized as following
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        
    path = os.path.join(test_dir, f'features_{fps}fps')
    speech_path = os.path.join(path, f'{speech_feat}')

    if not os.path.exists(path):
        os.makedirs(path)
        
    # speech features
    if not os.path.exists(speech_path):
        print('Processing speech features...')
        os.makedirs(speech_path)
        extract_melspec(audiopath, files, speech_path, fps)
    else:
        print('Found speech features. skipping processing...')    
    
    # Create test dataset
    print("Preparing datasets...")
        
    test_ctrl = import_and_pad(files, speech_path)
    
    ctrl_scaler = jl.load(os.path.join(processed_dir, 'input_scaler.sav'))
    test_ctrl = standardize(test_ctrl, ctrl_scaler)
    
    np.savez(os.path.join(test_dir,f'test_input_{fps}fps.npz'), clips = test_ctrl)
    copyfile(os.path.join(processed_dir, f'data_pipe_{fps}fps.sav'), os.path.join(test_dir,f'data_pipe_{fps}fps.sav'))
    copyfile(os.path.join(processed_dir, 'input_scaler.sav'), os.path.join(test_dir,'input_scaler.sav'))
    copyfile(os.path.join(processed_dir, 'output_scaler.sav'), os.path.join(test_dir,'output_scaler.sav'))
