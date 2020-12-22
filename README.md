# MoGlow and StyleGestures
This repository contains code for reproducing the papers "[MoGlow: Probabilistic and controllable motion synthesis using normalising flows](https://arxiv.org/abs/1905.06598)" and "[Style-controllable speech-driven gesture synthesis using normalising flows](https://diglib.eg.org/handle/10.1111/cgf13946)". Parts of the code are based on [this Glow implementation](https://github.com/chaiyujin/glow-pytorch/) by GitHub user [chaiyujin](https://github.com/chaiyujin/).

Please watch the following videos for an introduction to the papers:
* MoGlow: [https://youtu.be/pe-YTvavbtA](https://youtu.be/pe-YTvavbtA)
* Style Gestures: [https://youtu.be/egf3tjbWBQE](https://youtu.be/egf3tjbWBQE)

There is also a separate [MoGlow project page](https://simonalexanderson.github.io/MoGlow/).

# Prerequisites
The conda environment `moglow` defined in 'environment.yml' contains the required dependencies.

# Data
### Locomotion (joint positions)
Our preprocessed version of the human locomotion data is [available here](https://kth.box.com/s/quh3rwwl2hedwo32cdg1kq7pff04fjdf). Download it to the 'data/locomotion' folder. The data is pooled from the [Edinburgh Locomotion MOCAP Database](https://bitbucket.org/jonathan-schwarz/edinburgh_locomotion_mocap_dataset), [CMU Motion Capture Database](http://mocap.cs.cmu.edu/), and [HDM05](http://resources.mpi-inf.mpg.de/HDM05/) datasets. Please see the included README file for licenses and citations.

### Locomotion (joint angles)
We additionally trained MoGlow on locomotion data with joint angles parameterised using exponential maps. This allows synthesising motion for skinned characters. Here, we pooled the [LaFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and [Kinematica](https://github.com/Unity-Technologies/Kinematica_Demo) datasets, retargeted the motion to a uniform skeleton using Motion Builder, and then converted the data to BVH format. Unfortunately we cannot provide the processed bvh files due to the dataset licences. To reproduce these models, please redo these steps and place the bvh files in a `data/locomotion_rot/source/bvh/<some_subset>` folder. Then run `python prepare_locomotion_datasets.py` from the 'data_processing' folder. Here, `<some_subset>` should be replaced by the motion subset of interest; for example, we put all locomotion-related sessions in a 'loco_only' folder and all motion except wall-climbing in an 'all' folder.

### Gestures
We used the [Trinity Speech-Gesture Dataset](http://trinityspeechgesture.scss.tcd.ie/) to train our StyleGesture models. Trinity College Dublin requires interested parties to sign a license agreement and receive approval before gaining access to that material, so we cannot host it here. Once you have received approval, our processed version of the data, with motion data converted to BVH format (joint angles) and synchronised to the audio, [can be downloaded here](https://trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/). This is the same data that was used for the [GENEA Challenge 2020](https://genea-workshop.github.io/2020/) Place the data in the 'data/GENEA/source' folder and then run `python prepare_gesture_datasets.py` from the 'data_processing' folder.

# Training
Edit the 'hparams/xxx.json' file to modify network and training parameters. Start training by running the following command:
```
python train_moglow.py <hparams> <dataset>
```

Example 1: For training a locomotion model (joint positions):
```
python train_moglow.py 'hparams/preferred/locomotion.json' locomotion
```
Example 2: For training a locomotion model (joint angles):
```
python train_moglow.py 'hparams/preferred/locomotion_rot.json' locomotion_rot
```
Example 3: For training a gesture model:
```
python train_moglow.py 'hparams/preferred/style_gestures.json' trinity
```

**Important note:** Although the code allows multi-GPU training, this is not supported as it leads to incorrect results.

# Synthesis
Output samples are generated at specified intervals during training. Sampling from a pre-trained model is done by specifying the path in the 'hparams/xxx.json' file and then running `python train_moglow.py <hparams> <dataset>`.

# Studentising flows
The code also includes an option to change the latent distribution from the standard Gaussian to a Student's *t* distribution. We found that this change, which we term "Studentising flows", provides more robust training (gradient clipping can be removed) and gives better likelihood on held-out data. Please see our INNF+ 2020 paper "[Robust model training and generalisation with Studentising flows](https://arxiv.org/pdf/2006.06599.pdf)" to read more.

# References
If you use our code or build on our method, please credit our publications:
```
@article{henter2020moglow,
  author = {Henter, Gustav Eje and Alexanderson, Simon and Beskow, Jonas},
  doi = {10.1145/3414685.3417836},
  journal = {ACM Transactions on Graphics},
  number = {4},
  pages = {236:1--236:14},
  publisher = {ACM},
  title = {{M}o{G}low: {P}robabilistic and controllable motion synthesis using normalising flows},
  volume = {39},
  year = {2020}
}

@article{alexanderson2020style,
  title = {Style-controllable speech-driven gesture synthesis using normalising flows},
  author = {Alexanderson, Simon and Henter, Gustav Eje and Kucherenko, Taras and Beskow, Jonas},
  journal = {Computer Graphics Forum},
  volume = {39},
  number = {2},
  pages = {487--496},
  year = {2020},
  url = {https://diglib.eg.org/handle/10.1111/cgf13946},
  doi = {10.1111/cgf.13946},
  publisher = {John Wiley \& Sons}
}
```

If you use Studentising flows, please cite the following paper: 
```
@inproceedings{alexanderson2020robust,
  author = {Alexanderson, Simon and Henter, Gustav Eje},
  booktitle = {Proceedings of the ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models},
  series = {INNF+'20},
  pages = {25:1--25:9},
  title = {Robust model training and generalisation with {S}tudentising flows},
  url = {https://arxiv.org/abs/2006.06599},
  volume = {2},
  year = {2020}
}
```
