# StyleGestures
This repository contains code for reproducing the papers ["Moglow: Probabilistic and controllable motion synthesis using normalising flows"](https://arxiv.org/abs/1905.06598) and ["Style-Controllable Speech-Driven Gesture Synthesis Using Normalising Flows"](https://diglib.eg.org/handle/10.1111/cgf13946)

Please watch the following videos for an introduction:
Moglow: (https://youtu.be/ozVldUcFjZg)
Style Gestures: (https://youtu.be/egf3tjbWBQE)


# Prerequisites
The 'environment.yml' contains the required dependencies.

# Data
Our preprocessed version of the human locomotion data is available at https://kth.app.box.com/folder/116440954250. Download it to the 'data/locomotion' folder. The data is pooled from the Edinburgh Locomotion, CMU and HDM05 datasets. Please see the included README file for licenses and citations.

The gesture data is available at http://trinityspeechgesture.scss.tcd.ie/. Trinity College Dublin require interested parties to sign a license agreement and receive approval before gaining access the material, so we cannot host it here. We are looking to provide preprocessing guidelines and code in the near future.

# Training
Edit the 'hparams/xxx.json' file to modify network and traning parameters. Start training by running the following command:

```
python train_moglow.py <hparams> <dataset>
```

Example 1. For locomotion synthesis:
```
python train_moglow.py 'hparams/locomotion.json' locomotion
```
Example 2. For gesture synthesis:
```
python train_moglow.py 'hparams/style_gestures.json' trinity
```

# Inference
Output samples are generated at specified intervals during training. Inference from a pre-trained model is done by specifying the path in the 'hparams/xxx.json' file and then running `python train_moglow.py <hparams> <dataset>`. 

# References

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
  publisher={John Wiley & Sons}
}

```
