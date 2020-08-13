# StyleGestures
This repository contains code for reproducing the papers "[MoGlow: Probabilistic and controllable motion synthesis using normalising flows](https://arxiv.org/abs/1905.06598)" and "[Style-controllable speech-driven gesture synthesis using normalising flows](https://diglib.eg.org/handle/10.1111/cgf13946)". Parts of the code are based on the Glow implementation at https://github.com/chaiyujin/glow-pytorch/.

Please watch the following videos for an introduction to the papers:
* MoGlow: (https://youtu.be/ozVldUcFjZg)
* Style Gestures: (https://youtu.be/egf3tjbWBQE)

# Prerequisites
The conda environment `moglow` defined in 'environment.yml' contains the required dependencies.

# Data
Our preprocessed version of the human locomotion data is available at https://kth.box.com/s/quh3rwwl2hedwo32cdg1kq7pff04fjdf. Download it to the 'data/locomotion' folder. The data is pooled from the Edinburgh Locomotion, CMU Motion Capture, and HDM05 datasets. Please see the included README file for licenses and citations.

The gesture data is available at http://trinityspeechgesture.scss.tcd.ie/. Trinity College Dublin requires interested parties to sign a license agreement and receive approval before gaining access the material, so we cannot host it here. Preprocessing guidelines and code can be found in the 'data_processing' folder. There are still a few manual steps to get the data into the required bvh format. Please read the included README file for instructions.

# Training
Edit the 'hparams/xxx.json' file to modify network and training parameters. Start training by running the following command:
```
python train_moglow.py <hparams> <dataset>
```

Example 1. For training a locomotion model:
```
python train_moglow.py 'hparams/locomotion.json' locomotion
```
Example 2. For training a gesture model:
```
python train_moglow.py 'hparams/style_gestures.json' trinity
```

Note: Although the code allows multi-GPU training, this is not supported as it leads to incorrect results.

# Synthesis
Output samples are generated at specified intervals during training. Sampling from a pre-trained model is done by specifying the path in the 'hparams/xxx.json' file and then running `python train_moglow.py <hparams> <dataset>`.

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
