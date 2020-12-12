import numpy as np

import glob
import os
import sys
from shutil import copyfile
from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.writers import *
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib as jl

def inv_standardize(data, scaler):      
    shape = data.shape
    flat = data.reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled        

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
    
def create_synth_test_data(n_frames, nFeats):

    synth_data = np.zeros((16,n_frames,nFeats))
    lo_vel = 3.5
    hi_vel = 8
    max_vel = 15
    lo_r_vel = 0.08
    hi_r_vel = 0.11
    
    # straight line
    synth_data[1,:,-2] = lo_vel
    synth_data[2,:,-2] = hi_vel

    # circular paths 
    synth_data[3,:,-2] = lo_vel
    synth_data[3,:,-1] = hi_r_vel
    
    synth_data[4,:,-2] = hi_vel
    synth_data[4,:,-1] = lo_r_vel

    synth_data[5,:,-2] = lo_vel
    synth_data[5,:,-1] = lo_r_vel

    synth_data[6,:,-2] = hi_vel
    synth_data[6,:,-1] = hi_r_vel

    # sinusiodal paths
    angs = np.linspace(1,n_frames/100*2*np.pi,n_frames)    
    
    synth_data[7,:,-2] = lo_vel    
    synth_data[7,:,-1] = lo_r_vel*np.sin(angs)

    synth_data[8,:,-2] = lo_vel    
    synth_data[8,:,-1] = hi_r_vel*np.sin(angs)

    synth_data[9,:,-2] = hi_vel    
    synth_data[9,:,-1] = lo_r_vel*np.sin(angs)

    synth_data[10,:,-2] = hi_vel    
    synth_data[10,:,-1] = hi_r_vel*np.sin(angs)

    # sinusiodal speed
    synth_data[11,:,-2] = max_vel*0.5*(1+np.sin(angs))
    synth_data[12,:,-2] = max_vel*0.5*(1+np.sin(angs))
    synth_data[12,:,-1] = hi_r_vel*np.sin(angs)
    synth_data[13,:,-2] = hi_vel*0.5*(1+np.sin(angs))
    synth_data[13,:,-1] = hi_r_vel*np.sin(angs)
    synth_data[14,:,-2] = lo_vel*0.5*(1+np.sin(angs))
    synth_data[14,:,-1] = hi_r_vel*np.sin(angs)
    synth_data[15,:,-1] = hi_r_vel*np.sin(angs)
    
    #synth_data = standardize(synth_data, scaler)
    return synth_data.astype(np.float32)

def extract_features(bvh_dir, files, destpath, fps, fullbody=False, smooth_pos=4, smooth_rot=4):
    p = BVHParser()

    data_all = list()
    print("Importing data...")
    for f in files:
        ff = os.path.join(bvh_dir, f + '.bvh')
        print(ff)
        data_all.append(p.parse(ff))


    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=20)),
        ('mir', Mirror(axis='X', append=True)),
        ('rev', ReverseTime(append=True)),
    #    ('jtsel', JointSelector(['LowerBack','Spine','Spine1','Neck','Neck1','Head','RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand','RightHipJoint','RightUpLeg', 'RightLeg', 'RightFoot','LeftHipJoint','LeftUpLeg', 'LeftLeg', 'LeftFoot'], include_root=True)),
        ('jtsel', JointSelector(['Spine','Spine1','Neck','Head','RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot'], include_root=True)),
        ('root', RootTransformer('pos_rot_deltas', position_smoothing=4, rotation_smoothing=4)),
        ('exp', MocapParameterizer('expmap')), 
        ('cnst', ConstantsRemover()),
        ('npf', Numpyfier())
    ])


    print("Processing...")
    out_data = data_pipe.fit_transform(data_all)
    inv_data = data_pipe.inverse_transform(out_data)
    
    # the datapipe will append the mirrored files to the end
    assert len(out_data) == 4*len(files)
    
    jl.dump(data_pipe, os.path.join(destpath, 'data_pipe.sav'))
        
    fi=0
    for f in files:
        ff = os.path.join(destpath, f)
        print(ff)
        np.savez(ff + ".npz", clips=out_data[fi])
        np.savez(ff + "_mirrored.npz", clips=out_data[len(files)+fi])
        np.savez(ff + "_timerev.npz", clips=out_data[2*len(files)+fi])
        np.savez(ff + "_timerev_mirrored.npz", clips=out_data[3*len(files)+fi])
        
        # writer = BVHWriter()
        # with open(ff + '_inv.bvh','w') as f:
            # writer.write(inv_data[fi],f)
            
        fi=fi+1


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
            
def import_and_slice(files, motion_path, slice_window, slice_overlap, start=0, end=None):
    """Imports all features and slices them to samples with equal lenth time [samples, timesteps, features]."""
                    
    fi=0
    for file in files:
        #print(file)
        
        # slice dataset
        motion_data = np.load(os.path.join(motion_path, file + '.npz'))['clips'].astype(np.float32)    
        if motion_data.shape[0]<slice_window:
            #import pdb;pdb.set_trace()
            print("Too few frames in file: " + str(file))
            continue
        
        sliced = slice_data(motion_data, slice_window, slice_overlap)                        
        
        #if np.sum(sliced[:,:,18*3]>2)>0:
            # this is a heuristic filtering of movements with very high rotations, such as tumbling on the ground
            # print(file + " has rotation issues")
            
        
        if fi==0:
            out_data = sliced
        else:
            out_data = np.concatenate((out_data, sliced), axis=0)
            
        #print("out_data.shape:" + str(out_data.shape))
        #import pdb;pdb.set_trace()
        fi=fi+1

    return out_data[:,:,:-3], out_data[:,:,-3:]
    
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
        print("usage: python prepare_datasets.py")
        sys.exit(0)
     
    # Hardcoded preprocessing params and file structure. 
    # Modify these if you want the data in some different format
    train_window_secs = 5
    test_window_secs = 10
    window_overlap = 0.3
    fps = 20

    data_root = '../data/locomotion_rot/source/bvh'
    processed_dir = '../data/locomotion_rot/processed/'
    dataset = 'loco_only'
    bvhpath = os.path.join(data_root, dataset)
    held_out = ['Circles_Sprint_2', 'aiming2_subject5', 'Circles_Walk_2', 'run1_subject2'] 
        
    bvhfiles = []
    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(bvhpath):
        for file in f:
            if '.bvh' in file:
                ff=os.path.join(r, file)
                basename = os.path.splitext(os.path.basename(ff))[0]
                bvhfiles.append(basename)

    #print(bvhfiles)
    motion_feat = 'joint_rot'
    
    # processed data will be organized as following
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    path = os.path.join(processed_dir, f'features_{fps}fps')
    dataset_path = os.path.join(path, dataset)
    motion_path = os.path.join(dataset_path, motion_feat)

    if not os.path.exists(path):
        os.makedirs(path)
            
    # motion features
    if not os.path.exists(motion_path):
        print('Processing motion features...')
        os.makedirs(motion_path)
        # full body joint angles
        extract_features(bvhpath, bvhfiles, motion_path, fps, fullbody=True, smooth_pos=4, smooth_rot=4)
    else:
        print('Found motion features. skipping processing...')
    
    # copy pipeline for converting motion features to bvh
    copyfile(os.path.join(motion_path, 'data_pipe.sav'), os.path.join(processed_dir,f'data_pipe_{fps}fps.sav'))
           
    
    # divide in train, val, dev and test sets. Note that val and dev contains the same data allthough sliced in different ways.
    # - val data is used for logging and sliced the same way as the training data 
    # - dev data is sliced in longer sequences and used for visualization (we found that shorter snippets are hard to subjectivly evaluate)
    print("Preparing datasets...")
    
    files = []    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(motion_path):
        for file in f:
            if '.npz' in file:
                ff=os.path.join(r, file)
                basename = os.path.splitext(os.path.basename(ff))[0]
                files.append(basename)

    print(files)
    
    train_files = [f for f in files if f not in held_out]
    
    slice_win_train = train_window_secs*fps
    slice_win_test = test_window_secs*fps
    val_test_split = 20*test_window_secs*fps # 10 
    
    train_motion, train_ctrl = import_and_slice(train_files, motion_path, slice_win_train, window_overlap)
    val_motion, val_ctrl = import_and_slice(held_out, motion_path, slice_win_train, window_overlap, start=0, end=val_test_split)

    # the following sets are cut into longer clips without overlap. These are used for subjective evaluations during tuning (dev) and evaluation (test)
    dev_motion, dev_ctrl = import_and_slice(held_out, motion_path, slice_win_test, 0, start=0, end=val_test_split)
    test_motion, test_ctrl = import_and_slice(held_out, motion_path, slice_win_test, 0, start=val_test_split) # we only use the ctrl when testing. the test motion is used as ground truth
    synth_ctrl = create_synth_test_data(slice_win_test, test_ctrl.shape[2])
    synth_motion = np.zeros((synth_ctrl.shape[0], slice_win_test, test_motion.shape[2])).astype(np.float32) # dummy values that will never be used 
    concat_test_motion = np.concatenate((synth_motion, test_motion), axis=0)
    concat_test_ctrl = np.concatenate((synth_ctrl, test_ctrl), axis=0)
    
    train_motion, m_scaler = fit_and_standardize(train_motion)
    np.savez(os.path.join(processed_dir,'output_scaler.npz'), stds=m_scaler.scale_, means=m_scaler.mean_)
    train_ctrl, c_scaler = fit_and_standardize(train_ctrl)
    np.savez(os.path.join(processed_dir,'input_scaler.npz'), stds=c_scaler.scale_, means=c_scaler.mean_)
                            
    jl.dump(c_scaler, os.path.join(processed_dir,f'input_scaler.sav'))         
    jl.dump(m_scaler, os.path.join(processed_dir,f'output_scaler.sav'))         
    
    #import pdb;pdb.set_trace()
    np.savez(os.path.join(processed_dir,f'train_output_{fps}fps.npz'), clips = train_motion)
    np.savez(os.path.join(processed_dir,f'train_input_{fps}fps.npz'), clips = train_ctrl)
    np.savez(os.path.join(processed_dir,f'val_output_{fps}fps.npz'), clips = standardize(val_motion, m_scaler))
    np.savez(os.path.join(processed_dir,f'val_input_{fps}fps.npz'), clips = standardize(val_ctrl, c_scaler))
    np.savez(os.path.join(processed_dir,f'dev_output_{fps}fps.npz'), clips = standardize(dev_motion, m_scaler))
    np.savez(os.path.join(processed_dir,f'dev_input_{fps}fps.npz'), clips = standardize(dev_ctrl, c_scaler))
    np.savez(os.path.join(processed_dir,f'test_output_{fps}fps.npz'), clips = standardize(concat_test_motion, m_scaler))
    np.savez(os.path.join(processed_dir,f'test_input_{fps}fps.npz'), clips = standardize(concat_test_ctrl, c_scaler))
