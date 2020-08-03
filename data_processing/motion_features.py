import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle
import itertools
from datetime import datetime

import glob
import os
import sys
    
from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.writers import *
import joblib as jl

def nan_smooth(data, filt_len):

    win=filt_len//2
    
    nframes = len(data)
    out = np.zeros(data.shape)
    for i in range(nframes):
        if i < win:
            st = 0
        else:
            st=i-win
        if i > nframes-win-1:
            en = nframes
        else:
            en = i+win+1
        out[i,:] = np.nanmean(data[st:en,:], axis=0)
    return out

def extract_joint_angles(bvh_dir, files, destpath, fps, fullbody=False):
    p = BVHParser()

    data_all = list()
    print("Importing data...")
    for f in files:
        ff = os.path.join(bvh_dir, f + '.bvh')
        print(ff)
        data_all.append(p.parse(ff))

    if fullbody:
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
            ('mir', Mirror(axis='X', append=True)),
            ('jtsel', JointSelector(['Spine','Spine1','Neck','Head','RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'], include_root=True)),
            ('root', RootTransformer('pos_rot_deltas', position_smoothing=5, rotation_smoothing=10)),
            ('exp', MocapParameterizer('expmap')), 
            ('cnst', ConstantsRemover()),
            ('npf', Numpyfier())
        ])
    else:
        data_pipe = Pipeline([
           ('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
           ('root', RootTransformer('hip_centric')),
           ('mir', Mirror(axis='X', append=True)),
           ('jtsel', JointSelector(['Spine','Spine1','Spine2','Spine3','Neck','Neck1','Head','RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'], include_root=True)),
           ('exp', MocapParameterizer('expmap')), 
           ('cnst', ConstantsRemover()),
           ('np', Numpyfier())
        ])


    print("Processing...")
    out_data = data_pipe.fit_transform(data_all)
    
    # the datapipe will append the mirrored files to the end
    assert len(out_data) == 2*len(files)
    
    jl.dump(data_pipe, os.path.join(destpath, 'data_pipe.sav'))
        
    fi=0
    for f in files:
        ff = os.path.join(destpath, f)
        print(ff)
        np.savez(ff + ".npz", clips=out_data[fi])
        np.savez(ff + "_mirrored.npz", clips=out_data[len(files)+fi])
        fi=fi+1

def extract_hand_pos(bvh_dir, files, destpath, fps):

    p = BVHParser()

    data_all = list()
    for f in files:
        ff = os.path.join(bvh_dir, f + '.bvh')
        print(ff)
        data_all.append(p.parse(ff))
        
    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
        ('root', RootTransformer('hip_centric')),
        ('pos', MocapParameterizer('position')), 
        ('jtsel', JointSelector(['RightHand','LeftHand'], include_root=False)),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    
    assert len(out_data) == len(files)
    
    fi=0
    for f in files:
        ff = os.path.join(destpath, f)
        print(ff)
        np.save(ff + ".npy", out_data[fi])
        fi=fi+1

def extract_style_features(hand_pos_dir, files, destpath, fps, average_secs):
    filt_len = int(fps*average_secs)

    for f in files:
    
        ff = os.path.join(hand_pos_dir, f + '.npy')
        data = np.load(ff)
        cols = [1,4]
        
        # Extract the average hand heights over the specified time period
        h = data[:,cols]
        h_sm = nan_smooth(h, filt_len)
        np.save(os.path.join(os.path.join(destpath, 'MG-H'), f) + '.npy', h_sm[:,0])
        #np.save(os.path.join(os.path.join(destpath, 'MG-H'), f) + '.npy', h_sm[:,1])

        #mirrored
        np.save(os.path.join(os.path.join(destpath, 'MG-H'), f) + '_mirrored.npy', h_sm[:,1])
        #np.save(os.path.join(os.path.join(destpath, 'MG-H'), f) + '_mirrored.npy', h_sm[:,0])

        
        # Extract the average hands velocity over the specified time period
        rcols = [0,1,2] 
        lcols = [3,4,5] 
        rr=data[:,rcols]
        ll=data[:,lcols]
        dr=np.linalg.norm(np.diff(rr, axis=0), axis=1)
        dl=np.linalg.norm(np.diff(ll, axis=0), axis=1)
        dl_sm = nan_smooth(dl[:,None], filt_len)
        dr_sm = nan_smooth(dr[:,None], filt_len)
        speed = dr_sm + dl_sm
        np.save(os.path.join(os.path.join(destpath, 'MG-V'), f) + '.npy', speed)
        np.save(os.path.join(os.path.join(destpath, 'MG-V'), f) + '_mirrored.npy', speed) # same
        
        # Extract the hand radius
        rcols = [0,2] 
        lcols = [3,5] 
        rr=data[:,rcols]
        ll=data[:,lcols]
        dr=np.linalg.norm(rr, axis=1)
        dl=np.linalg.norm(ll, axis=1)
        dl_sm = nan_smooth(dl[:,None], filt_len)
        dr_sm = nan_smooth(dr[:,None], filt_len)        
        radius = dr_sm + dl_sm
        np.save(os.path.join(os.path.join(destpath, 'MG-R'), f) + '.npy', radius)
        np.save(os.path.join(os.path.join(destpath, 'MG-R'), f) + '_mirrored.npy', radius)


