import numpy as np
import librosa
import soundfile as sf

import glob
import os
import sys
import csv
import subprocess

from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.writers import *

def uniquify(file_name):
    if os.path.isfile(file_name):
        expand = 1
        while True:
            expand += 1
            spl = os.path.splitext(file_name)
            new_file_name = spl[0] + '_' + str(expand) + spl[1]
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break
    return file_name

def process_bvh(basepath, destpath, filename, start, stop):
    p = BVHParser()
    print(f'Processing BVH {filename} from: {start} to {stop}')
    outpath = os.path.join(destpath,'bvh')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outfile = uniquify(os.path.join(outpath,filename + '.bvh'))
    infile = os.path.join(os.path.join(basepath,'bvh'), filename + '.bvh')
    bvh_data = p.parse(infile, start=start, stop=stop)
    dwnsampl = DownSampler(tgt_fps=60)
    out_data = dwnsampl.fit_transform([bvh_data])
    writer = BVHWriter()
    with open(outfile,'w') as f:
        writer.write(out_data[0], f)
    
def process_wav_mic(basepath, destpath, filename, starttime, endtime):
    print(f'Processing MICROPHONE AUDIO {filename} from: {starttime} to {endtime}')
    infile = os.path.join(os.path.join(basepath,'audio'), filename + '.wav')
    outpath = os.path.join(destpath,'audio')
    X,fs = librosa.load(infile, sr=48000)
    start_idx = int(np.round(starttime*fs))
    if(endtime>0):
        end_idx = int(np.round(endtime*fs))
    else:
        end_idx = -1
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outfile = uniquify(os.path.join(outpath,filename + '.wav'))
    sf.write(outfile, X[start_idx:end_idx], fs)
    
if __name__ == "__main__":        
    basepath = '../data/trinity_new/unsynchronized/'
    destpath = '../data/trinity_new/source/'
    offset_file = 'offsets.csv'
    if not os.path.exists(destpath):
        os.makedirs(destpath)

    with open(offset_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            filename = row[0]
            start_frame = int(row[1])
            stop_frame = int(row[2])
            offset_bvh = int(row[4])
            fps_bvh = int(row[5])
            offset_wav = float(row[6])
            start_time = float(start_frame)/30.0 
            stop_time = float(stop_frame)/30.0

            start_bvh = int(start_frame*(fps_bvh/30.0))-offset_bvh
            start_wav = start_time+offset_wav
            if stop_frame>0:
                stop_bvh = int(stop_frame*(fps_bvh/30.0))-offset_bvh
                stop_wav = stop_time+offset_wav
            else:
                stop_bvh = -1
                stop_wav = -1

            
            process_bvh(basepath, destpath, filename, start_bvh, stop_bvh)
            process_wav_mic(basepath, destpath, filename, start_wav, stop_wav)
