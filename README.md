# StyleGestures
This repository contains code for reproducing the papers "Moglow: Probabilistic and controllable motion synthesis using normalising flows" (https://arxiv.org/abs/1905.06598) and "Style-Controllable Speech-Driven Gesture Synthesis Using Normalising Flows" (https://diglib.eg.org/handle/10.1111/cgf13946)
[Simon Alexanderson](https://www.kth.se/profile/simonal?l=en), [Gustav Eje Henter](https://people.kth.se/~ghe/), [Taras Kucherenko](https://svito-zar.github.io/), [Jonas Beskow](https://www.kth.se/profile/beskow?l=en)

[Demo video](https://youtu.be/egf3tjbWBQE)

# Prerequisites
The 'environment.yml' contains the required dependencies.

# Training
Edit the 'hparams/xxx.json' file to modify network and traning parameters. Start train by running the following command:

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
  title={Moglow: Probabilistic and controllable motion synthesis using normalising flows},
  author={Henter, Gustav Eje and Alexanderson, Simon and Beskow, Jonas},
  journal={arXiv preprint arXiv:1905.06598},
  year={2019}
}    

@inproceedings{alexanderson2020style,
  title={Style-Controllable Speech-Driven Gesture Synthesis Using Normalising Flows},
  author={Alexanderson, Simon and Henter, Gustav Eje and Kucherenko, Taras and Beskow, Jonas},
  booktitle={EUROGRAPHICS 2020},
  year={2020}
}
```
