# StyleGestures
This repository contains code for reproducing the papers "Moglow: Probabilistic and controllable motion synthesis using normalising flows" (https://arxiv.org/abs/1905.06598) and "Style-Controllable Speech-Driven Gesture Synthesis Using Normalising Flows" (https://diglib.eg.org/handle/10.1111/cgf13946)

# To run the code
`python train_moglow.py <hparams> <dataset>`

For locomotion synthesis:
```
python train_moglow.py 'hparams/locomotion.json' locomotion
```
For gesture synthesis:
```
python train_moglow.py 'hparams/style_gestures.json' trinity
```
