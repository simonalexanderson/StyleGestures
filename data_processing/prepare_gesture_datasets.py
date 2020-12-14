import numpy as np

import glob
import os
import sys
from shutil import copyfile
from motion_features import extract_joint_angles, extract_hand_pos, extract_style_features
from audio_features import extract_melspec
import scipy.io.wavfile as wav
from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.writers import *
from sklearn.preprocessing import StandardScaler
import joblib as jl

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
    
def cut_audio(filename, timespan, destpath, starttime=0.0, endtime=-1.0):
    print(f'Cutting AUDIO {filename} into intervals of {timespan}')
    fs,X = wav.read(filename)
    if endtime<=0:
        endtime = len(X)/fs
    suffix=0
    while (starttime+timespan) <= endtime:
        out_basename = os.path.splitext(os.path.basename(filename))[0]
        wav_outfile = os.path.join(destpath, out_basename + "_" + str(suffix).zfill(3) + '.wav')
        start_idx = int(np.round(starttime*fs))
        end_idx = int(np.round((starttime+timespan)*fs))+1
        if end_idx >= X.shape[0]:
            return
        wav.write(wav_outfile, fs, X[start_idx:end_idx])
        starttime += timespan
        suffix+=1

def cut_bvh(filename, timespan, destpath, starttime=0.0, endtime=-1.0):
    print(f'Cutting BVH {filename} into intervals of {timespan}')
    
    p = BVHParser()
    bvh_data = p.parse(filename)
    if endtime<=0:
        endtime = bvh_data.framerate*bvh_data.values.shape[0]

    writer = BVHWriter()
    suffix=0
    while (starttime+timespan) <= endtime:
        out_basename = os.path.splitext(os.path.basename(filename))[0]
        bvh_outfile = os.path.join(destpath, out_basename + "_" + str(suffix).zfill(3) + '.bvh')
        start_idx = int(np.round(starttime/bvh_data.framerate))
        end_idx = int(np.round((starttime+timespan)/bvh_data.framerate))+1
        if end_idx >= bvh_data.values.shape[0]:
            return
            
        with open(bvh_outfile,'w') as f:
            writer.write(bvh_data, f, start=start_idx, stop=end_idx)
            
        starttime += timespan
        suffix+=1
        
def slice_data(data, window_size, overlap):

    nframes = data.shape[0]
    overlap_frames = (int)(overlap*window_size)
    
    n_sequences = (nframes-overlap_frames)//(window_size-overlap_frames)
    sliced = np.zeros((n_sequences, window_size, data.shape[1])).astype(np.float32)
    
    if n_sequences>0:

        # extract sequences from the data
        for i in range(0,n_sequences):
            frameIdx = (window_size-overlap_frames) * i
            sliced[i,:,:] = data[frameIdx:frameIdx+window_size,:].copy()
    else:
        print("WARNING: data too small for window")
                    
    return sliced
    
def align(data1, data2):
    """Truncates to the shortest length and concatenates"""
    
    nframes1 = data1.shape[0]
    nframes2 = data2.shape[0]
    if nframes1<nframes2:
        return np.concatenate((data1, data2[:nframes1,:]), axis=1)
    else:
        return np.concatenate((data1[:nframes2,:], data2), axis=1)
        
def import_data(file, motion_path, speech_data, style_path, mirror=False, start=0, end=None):
    """Loads a file and concatenate all features to one [time, features] matrix. 
     NOTE: All sources will be truncated to the shortest length, i.e. we assume they
     are time synchronized and has the same start time."""
    
    suffix=""
    if mirror:
        suffix="_mirrored"
        
    motion_data = np.load(os.path.join(motion_path, file + suffix + '.npz'))['clips'].astype(np.float32)        
    n_motion_feats = motion_data.shape[1]

    speech_data = np.load(os.path.join(speech_path, file + '.npy')).astype(np.float32)
    
    if style_path is not None:
        style_data = np.load(os.path.join(style_path, file + suffix + '.npy')).astype(np.float32)
        control_data = align(speech_data,style_data[:])
    else:
        control_data = speech_data
        
    concat_data = align(motion_data, control_data)
    
    if not end:
        end = concat_data.shape[0]
        
    return concat_data[start:end,:], n_motion_feats

def import_and_slice(files, motion_path, speech_data, style_path, slice_window, slice_overlap, mirror=False, start=0, end=None):
    """Imports all features and slices them to samples with equal lenth time [samples, timesteps, features]."""
                    
    fi=0
    for file in files:
        print(file)
        
        # slice dataset
        concat_data, n_motion_feats = import_data(file, motion_path, speech_data, style_path, False, start, end)        
        sliced = slice_data(concat_data, slice_window, slice_overlap)                
        
        if mirror:
            concat_mirr, nmf = import_data(file, motion_path, speech_data, style_path, True, start, end)
            sliced_mirr = slice_data(concat_mirr, slice_window, slice_overlap)
            
            # append to the sliced dataset
            sliced = np.concatenate((sliced, sliced_mirr), axis=0)
        
        if fi==0:
            out_data = sliced
        else:
            out_data = np.concatenate((out_data, sliced), axis=0)
        fi=fi+1

    return out_data[:,:,:n_motion_feats], out_data[:,:,n_motion_feats:]
    
if __name__ == "__main__":
    '''
    Converts bvh and wav files into features, slices in equal length intervals and divides the data
    into training, validation and test sets. Adding an optional style argument ("MG-R", "MG-V", "MG-H" or "MS-S") 
    will add features for style control.
    '''     
    if len(sys.argv)==1:
        style_path = None
    elif len(sys.argv)==2:
        style_path = sys.argv[1]
    else:
        print("usage: python prepare_datasets.py [MS-S|MG-R|MG-V|MG-H]")
        sys.exit(0)
     
    # Hardcoded preprocessing params and file structure. 
    # Modify these if you want the data in some different format
    train_window_secs = 6
    test_window_secs = 20
    window_overlap = 0.5
    fps = 20

    data_root = '../data/GENEA/source'
    bvhpath = os.path.join(data_root, 'bvh')
    audiopath = os.path.join(data_root, 'audio')
    held_out = ['Recording_008']
    processed_dir = '../data/GENEA/processed'    
    
    files = []
    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(bvhpath):
        for file in sorted(f):
            if '.bvh' in file:
                ff=os.path.join(r, file)
                basename = os.path.splitext(os.path.basename(ff))[0]
                files.append(basename)

    print(files)
    motion_feat = 'joint_rot'
    speech_feat = 'melspec'
    
    # processed data will be organized as following
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    path = os.path.join(processed_dir, f'features_{fps}fps')
    motion_path = os.path.join(path, f'{motion_feat}')
    speech_path = os.path.join(path, f'{speech_feat}')
    hand_path = os.path.join(path, 'hand_pos')
    vel_path = os.path.join(path, 'MG-V')
    radius_path = os.path.join(path,'MG-R')
    rh_path = os.path.join(path, 'MG-H')
    #lh_path = os.path.join(path, 'MG-LH')
    sym_path = os.path.join(path, 'MG-S')

    if not os.path.exists(path):
        os.makedirs(path)
        
    # speech features
    if not os.path.exists(speech_path):
        print('Processing speech features...')
        os.makedirs(speech_path)
        extract_melspec(audiopath, files, speech_path, fps)
    else:
        print('Found speech features. skipping processing...')
    
    # upper body joint angles
    if not os.path.exists(motion_path):
        print('Processing motion features...')
        os.makedirs(motion_path)
        extract_joint_angles(bvhpath, files, motion_path, fps, fullbody=False)
        # full body joint angles
        #extract_joint_angles(bvhpath, files, motion_path, fps, fullbody=True, smooth_pos=5, smooth_rot=10)
    else:
        print('Found motion features. skipping processing...')
    
    # copy pipeline for converting motion features to bvh
    copyfile(os.path.join(motion_path, 'data_pipe.sav'), os.path.join(processed_dir,f'data_pipe_{fps}fps.sav'))
   
    # optional style features    
    if not os.path.exists(hand_path):
        print('Processing style features...')
        os.makedirs(hand_path)
        os.makedirs(vel_path)
        os.makedirs(radius_path)
        os.makedirs(rh_path)
        #os.makedirs(lh_path)
        os.makedirs(sym_path)
        extract_hand_pos(bvhpath, files, hand_path, fps)
        extract_style_features(hand_path, files, path, fps, average_secs=4)
    else:
        print('Found style features. skipping processing...')
        
    
    # divide in train, val, dev and test sets. Note that val and dev contains the same data allthough sliced in different ways.
    # - val data is used for logging and sliced the same way as the training data 
    # - dev data is sliced in longer sequences and used for visualization (we found that shorter snippets are hard to subjectivly evaluate)
    print("Preparing datasets...")
    
    train_files = [f for f in files if f not in held_out]
    
    slice_win_train = train_window_secs*fps
    slice_win_test = test_window_secs*fps
    val_test_split = 20*test_window_secs*fps # 10 
    
    train_motion, train_ctrl = import_and_slice(train_files, motion_path, speech_path, style_path, slice_win_train, window_overlap, mirror=True)
    val_motion, val_ctrl = import_and_slice(held_out, motion_path, speech_path, style_path, slice_win_train, window_overlap, mirror=True, start=0, end=val_test_split)

    # the following sets are cut into longer clips without overlap. These are used for subjective evaluations during tuning (dev) and evaluation (test)
    dev_motion, dev_ctrl = import_and_slice(held_out, motion_path, speech_path, style_path, slice_win_test, 0, mirror=False, start=0, end=val_test_split)
    test_motion, test_ctrl = import_and_slice(held_out, motion_path, speech_path, style_path, slice_win_test, 0, mirror=False, start=val_test_split)
    
    # if style controlled, set the control values to 15%, 50% and 85% quantiles
    if style_path is not None:
        dev_ctrl[0::3,:,-1].fill(np.quantile(train_ctrl[:,:,-1],0.15))
        dev_ctrl[1::3,:,-1].fill(np.quantile(train_ctrl[:,:,-1],0.5))
        dev_ctrl[2::3,:,-1].fill(np.quantile(train_ctrl[:,:,-1],0.85))
        test_ctrl[0::3,:,-1].fill(np.quantile(train_ctrl[:,:,-1],0.15))
        test_ctrl[1::3,:,-1].fill(np.quantile(train_ctrl[:,:,-1],0.5))
        test_ctrl[2::3,:,-1].fill(np.quantile(train_ctrl[:,:,-1],0.85))
                    
    #import pdb;pdb.set_trace()
    train_ctrl, input_scaler = fit_and_standardize(train_ctrl)
    train_motion, output_scaler = fit_and_standardize(train_motion)
    val_ctrl = standardize(val_ctrl, input_scaler)
    val_motion = standardize(val_motion, output_scaler)
    dev_ctrl = standardize(dev_ctrl, input_scaler)
    dev_motion = standardize(dev_motion, output_scaler)
    test_ctrl = standardize(test_ctrl, input_scaler)
    test_motion = standardize(test_motion, output_scaler)
        
    jl.dump(input_scaler, os.path.join(processed_dir,f'input_scaler.sav'))         
    jl.dump(output_scaler, os.path.join(processed_dir,f'output_scaler.sav'))         
    np.savez(os.path.join(processed_dir,f'train_output_{fps}fps.npz'), clips = train_motion)
    np.savez(os.path.join(processed_dir,f'train_input_{fps}fps.npz'), clips = train_ctrl)
    np.savez(os.path.join(processed_dir,f'val_output_{fps}fps.npz'), clips = val_motion)
    np.savez(os.path.join(processed_dir,f'val_input_{fps}fps.npz'), clips = val_ctrl)
    np.savez(os.path.join(processed_dir,f'dev_output_{fps}fps.npz'), clips = dev_motion)
    np.savez(os.path.join(processed_dir,f'dev_input_{fps}fps.npz'), clips = dev_ctrl)
    np.savez(os.path.join(processed_dir,f'test_output_{fps}fps.npz'), clips = test_motion)
    np.savez(os.path.join(processed_dir,f'test_input_{fps}fps.npz'), clips = test_ctrl)

    # finally prepare data for visualisation, i.e. the dev and test data in wav and bvh format    
    dev_vispath = os.path.join(processed_dir, 'visualization_dev')
    test_vispath = os.path.join(processed_dir, 'visualization_test')    
    if not os.path.exists(dev_vispath):
        os.makedirs(dev_vispath)
        os.makedirs(test_vispath)
        
        for file in held_out:
            cut_audio(os.path.join(audiopath, file + ".wav"), test_window_secs, dev_vispath, starttime=0.0,endtime=10*test_window_secs)
            cut_audio(os.path.join(audiopath, file + ".wav"), test_window_secs, test_vispath, starttime=10*test_window_secs)
    else:
        print('Found visualization data. Skipping processing...')
