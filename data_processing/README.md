PREPROCESSING GUIDLINES FOR GESTURE GENERATION
==============================================
The 'prepare_datasets.py' script extracts features and prepares training, validation and test datasets for StyleGestures.
Our required input format is synchronized motion (bvh) and audio (48k wav) files, which should be placed in the ~StyleGestures/data/trinity/source folder.

run 'python prepare_datasets.py' for our base system
run 'python prepare_datasets.py [MG-V|MG-H|MG-R|MG-S]' for style controlled systems

NOTE: Running the script repeatedly will overwrite previous datasets, so backup these if you want to switch between systems or train multiple systems in parallel.

NOTES on the trinity speech gesture (TSG) dataset (http://trinityspeechgesture.scss.tcd.ie)
===========================================================================================
- Unfortunately TSG is not ready for our preprocessing of-the-bat as it needs to be exported to bvh and synchronized first. We did a few steps manually using Autodesk MotionBuilder and Maya and there are no scripts automating this process. NOTE: You will need to redo these steps which requires a basic knowledge of Maya and Motionbuilder. Please do not ask us for guidance, there are tutorials online.

- As Trinity College Dublin requires interested parties to sign a license agreement and receive approval before gaining access the source data, we cannot share the data in our required (bvh and wav) format.

- The TSG skeleton does not outgo from a T-Pose (when setting all joint angles to zero). Outgoing from a T-pose is a good practice as it limits the risks for discontinuities in angle representatinons. We fixed this by first posing the skeleton in a T-Pose in Maya and freezing the joint angles, and the retargetting the motion to the new skeleton. This may or may not be important for the angle features.

- TSG motion data is in FBX format and needs to be exported to BVH. We did this in MotionBuilder.

- TSG audio and motion data are not synchronized, and needs to be cut according to specified start times. Timings are supplied in the dataset, but we found problems with sessions 21,27 and 30, which goes out-of-sync mid-session. Use our script 'syncronize_trinity.py', which splits these sessions into syncronized parts. This script also resamples all bvh files to the same frame rate of 60fps (TSG frame rate vary between 120 and 60 fps) and audio sampling rate (48k). 
