PREPROCESSING GUIDLINES FOR GESTURE GENERATION
==============================================
The ´prepare_gesture_datasets.py´ script extracts features and prepares training, validation and test datasets for StyleGestures.
Our required input format is synchronized motion (bvh) and audio (48k wav) files, which should be placed in the ~StyleGestures/data/GENEA/source folder.

To preprocess data for our base system:
```
python prepare_gesture_datasets.py 
```
For style controlled systems:
```
python prepare_gesture_datasets.py [MG-V|MG-H|MG-R|MG-S]
```

To prepare input data for other speakers than the trinity data, use the ´prepare_gesture_testdata.py´ and run it with a pretrained model. In these cases, gesture quality is generally lower.

NOTE: Running the script repeatedly will overwrite previous datasets, so make a backup if you want to switch between systems or train multiple systems in parallel. You can switch datasets in the hparams json file.

PREPROCESSING GUIDLINES FOR LOCOMOTION GENERATION WITH JOINT ANGLES
===================================================================
The ´prepare_locomotion_datasets.py´ script extracts features and prepares training, validation and test datasets for Moglow. Our required input format is bvh files, placed in the `data/locomotion_rot/source/bvh/<some_subset>` folder. Here, <some_subset> should be replaced, e.g. we put only the locomotion sessions in a 'loco_only' folder and all except wall-climbing in a 'all' folder.

To run:
```
python prepare_locomotion_datasets.py 
```
