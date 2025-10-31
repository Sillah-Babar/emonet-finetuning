# Estimation of Continuous Valence and Arousal Levels from Faces in Naturalistic Conditions

This project focuses on estimating **continuous valence and arousal (V–A) levels** from human facial expressions under naturalistic conditions.  
The workflow integrates **affective modeling**, **evaluation**, and a **real-time demonstration UI** built on top of *EmoNet* pretrained models.


---

## Model Application and Fine-tuning

### Emodataset fine-tuning (by Matt Stirling)

`emodataset-on-emonet/` contains a copy of the repo used for fine-tuning EmoNet on Emodataset, as well as the code for emonet model itself (unmodified, except to fix pylance warnings). It also contains some saved parameters, some csv's of fine-tuning and some graphics. Follow the [`README.md`](./emodataset-on-emonet/README.md) and be sure to run from the root of *that* directory. 






### UI and Demo (by Chengyi Su)

A **real-time interactive demo** showcases valence–arousal estimation directly from webcam input. You can check a showcase video here: [demo](https://www.youtube.com/watch?v=waxnplqpPgs)

**For demo details**, including setup instructions and usage guide, please refer to:  [`demo/README.md`](./demo/README.md) Be sure to run from the root of *that* directory. 

### EmoNet: Multi-label Emotion Recognition on EMOTIC Dataset (sillah babar)


A PyTorch implementation of EmoNet with architectural variations for multi-label emotion recognition, trained on the EMOTIC dataset. This model predicts 26 discrete emotions along with continuous valence and arousal values.
Overview
This repository provides:

Face visibility computation using MediaPipe Face Mesh
Data preprocessing with normalized valence/arousal values
Multi-label emotion classification with continuous affect prediction
Comprehensive evaluation and visualization tools
Pre-trained model weights

Please refer to tge Readme.md [`README.md`](./emotic-on-emonet/Readme.md) for more detailed instructions on how an inference.
