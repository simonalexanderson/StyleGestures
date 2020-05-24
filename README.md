# StyleGestures
===============
This repository contains code for reproducing the papers "Moglow: Probabilistic and controllable motion synthesis using normalising flows" and "Style-Controllable Speech-Driven Gesture Synthesis Using Normalising Flows"

#To run the code
===============
´python train_moglow.py <hparams> <dataset>´

For locomotion synthesis:
```
python train_moglow.py 'hparams/locomotion.json' locomotion
```
For gesture synthesis:
```
python train_moglow.py 'hparams/style_gestures.json' trinity
```
#To adapt to new data
====================
Modify the hyperparameters defined in `hparams/style_gestures.json` or `hparams/locomotion.json`. Note that the following entries must match the input and control dimensions:

```
"n_features": number of output features (joint coordinates in our case)
"cond_channels": size of the conditioning information. This is calculated as sequence_length * n_features + (sequence_length + 1 + n_lookahead) * n_control_features.
```
Example 1: In style_gestures.json:
```
sequence_length = 5
n_lookahead = 20
n_control_features = 27
n_features = 45 (for the upper body)
=> cond_channels = 45*5 + (5 + 1 + 20) * 27 = 927
```
Example 2: In locomtion.json:
```
sequence_length = 10
n_lookahead = 0
n_control_features = 3 (dx, dz, dr)
n_features = 63 (joint coordinates)
=> cond_channels = 63*10 + 11 * 3 = 663
```
Start tuning the network with dropout set to 0.5 and number of flows K set to 4 or 8. 
