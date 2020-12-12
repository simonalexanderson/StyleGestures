# StyleGestures
This repository contains code for reproducing the papers "[MoGlow: Probabilistic and controllable motion synthesis using normalising flows](https://arxiv.org/abs/1905.06598)" and "[Style-controllable speech-driven gesture synthesis using normalising flows](https://diglib.eg.org/handle/10.1111/cgf13946)". Parts of the code are based on the Glow implementation at https://github.com/chaiyujin/glow-pytorch/.

Please watch the following videos for an introduction to the papers:
* MoGlow: (https://youtu.be/ozVldUcFjZg)
* Style Gestures: (https://youtu.be/egf3tjbWBQE)

# Prerequisites
The conda environment `moglow` defined in 'environment.yml' contains the required dependencies.

# Data
Locomotion (joint positions): Our preprocessed version of the human locomotion data is available at https://kth.app.box.com/folder/116440954250. Download it to the `data/locomotion` folder. The data is pooled from the Edinburgh Locomotion, CMU Motion Capture, and HDM05 datasets. Please see the included README file for licenses and citations.

Locomotion (joint angles): We additionally trained the model with joint angles represented as exponential maps. This allows synthesising motion for skinned characters. Here, we pooled the [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and the [Kinematica](https://github.com/Unity-Technologies/Kinematica_Demo) datasets, retargeted the motion to a uniform skeleton (using Motion Builder), and then converted the data to bvh. Unfortunalty we cannot provide the processed bvh files due to the dataset licences. To reproduce, please follow the mentioned steps and download the bvh files to a `data/locomotion_rot/source/bvh/<some_subset>` folder and run `python prepare_locomotion_datasets.py` from the `data_processing` folder. Here, <some_subset> should be replaced, e.g. we put only the locomotion sessions in a 'loco_only' folder and all except wall-climbing in a 'all' folder.

Gestures: We used the [Trinity Speech Gesture dataset](http://trinityspeechgesture.scss.tcd.ie/) to train the model. Trinity College Dublin requires interested parties to sign a license agreement and receive approval before gaining access the material, so we cannot host it here. Our processed version of the data, with motion data converted to bvh format and synchronized to the audio, is available [HERE](https://trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/). Download it to the data/GENEA/source folder and run `python prepare_gesture_datasets.py` from the `data_processing` folder.

# Training
Edit the 'hparams/xxx.json' file to modify network and training parameters. Start training by running the following command:
```
python train_moglow.py <hparams> <dataset>
```

Example 1. For training a locomotion model (joint positions):
```
python train_moglow.py 'hparams/locomotion.json' locomotion
```
Example 2. For training a locomotion model (joint angles):
```
python train_moglow.py 'hparams/locomotion_rot.json' locomotion_rot
```
Example 3. For training a gesture model:
```
python train_moglow.py 'hparams/style_gestures.json' trinity
```

IMPORTANT NOTE! Although the code allows multi-GPU training, this is not supported as it leads to incorrect results.

# Synthesis
Output samples are generated at specified intervals during training. Sampling from a pre-trained model is done by specifying the path in the 'hparams/xxx.json' file and then running `python train_moglow.py <hparams> <dataset>`.

# Studentising flows
We also inculde an option to change the latent distribution from the standard Gaussian to a Student's t distribution. We found that this change, that we term 'Studentising flows', provides more robust training (gradient clipping can be removed) and generalises better to held-out validation data. Please see our INNF+ (ICML workshop on invertible neural networks) paper [Robust model training and generalisation with Studentising flows](https://arxiv.org/pdf/2006.06599.pdf).

# References
If you use our code or build on our method, please credit our publications:
```
@article{henter2019moglow,
  title={{M}o{G}low: {P}robabilistic and controllable motion synthesis using normalising flows},
  author={Henter, Gustav Eje and Alexanderson, Simon and Beskow, Jonas},
  journal={arXiv preprint arXiv:1905.06598},
  year={2019}
}

@article{alexanderson2020style,
  title={Style-controllable speech-driven gesture synthesis using normalising flows},
  author={Alexanderson, Simon and Henter, Gustav Eje and Kucherenko, Taras and Beskow, Jonas},
  journal={Computer Graphics Forum},
  volume={39},
  number={2},
  pages={487--496},
  year={2020},
  url={https://diglib.eg.org/handle/10.1111/cgf13946},
  doi={10.1111/cgf.13946},
  publisher={John Wiley \& Sons}
}
```

If you find Studentising flows usful for training, please cite the following paper 
```
@inproceedings{alexanderson2020robust,
  author={Alexanderson, Simon and Henter, Gustav Eje},
  booktitle=ProcINNF,
  series={INNF+'20},
  articleno={15},
  numpages={9},
  pages={25:1--25:9},
  title={Robust model training and generalisation with {S}tudentising flows},
  url={https://arxiv.org/abs/2006.06599},
  volume={2},
  year={2020}
}
```
