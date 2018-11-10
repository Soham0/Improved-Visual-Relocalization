# Improved Visual Relocalization by Discovering Anchor Points

This repository contains the code for the paper:

[Improved Visual Relocalization by Discovering Anchor Points](http://bmvc2018.org/contents/papers/0962.pdf)

[Soham Saha](https://soham0.github.io/), [Girish Varma](https://geevi.github.io/), [C.V.Jawahar](https://faculty.iiit.ac.in/~jawahar/)

## Citation

Please consider citing our work, if you find it useful in your research:

```
@article{sahaimproved,
  title={Improved Visual Relocalization by Discovering Anchor Points},
  author={Saha, Soham and Varma, Girish and Jawahar, CV}
}
```

## Introduction

We address the visual relocalization problem of predicting the location and camera
orientation or pose (6DOF) of the given input scene. We propose a method based on how
humans determine their location using the visible landmarks. We define anchor points
uniformly across the route map and propose a deep learning architecture which predicts
the most relevant anchor point present in the scene as well as the relative offsets with
respect to it. The relevant anchor point need not be the nearest anchor point to the ground
truth location, as it might not be visible due to the pose. Hence we propose a multi task
loss function, which discovers the relevant anchor point, without needing the ground truth
for it.

## Installation and Dependencies

- [Python 2.7](https://www.python.org/)
- [Pytorch 0.3.1 and TorchVision](https://pytorch.org/)

### Usage

A model is trained for every scene with different number of anchor points for every scene. The path to the scene and the parameters must be changed in the code.

This code assumes that each scene has a separate folder and is saved in the current path. The paths need to be modified accordingly.
```
python create_cambridge_scene.py
python preprocess_cambridge_scene.py
python localize_scene.py
```
The test performance needs to be computed from the saved files.

### Contact

Please contact us at :

sohamsaha\[dot]cs\[at]gmail.com
soham\[dot]saha\[at]research\[dot]iiit\[dot]ac\[dot]in
