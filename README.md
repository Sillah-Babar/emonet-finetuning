# Estimation of Continuous Valence and Arousal Levels from Faces in Naturalistic Conditions

This project focuses on estimating **continuous valence and arousal (V–A) levels** from human facial expressions under naturalistic conditions.  
The workflow integrates **affective modeling**, **evaluation**, and a **real-time demonstration UI** built on top of *EmoNet* pretrained models.


---

## Model Application and Fine-tuning

### Emodataset fine-tuning (by Matt Stirling)

`emodataset-on-emonet/` contains a copy of the repo used for fine-tuning EmoNet on Emodataset, as well as the code for emonet model itself (unmodified, except to fix pylance warnings). It also contains some saved parameters, some csv's of fine-tuning and some graphics. Follow the [`README.md`](./emodataset-on-emonet/README.md) and be sure to run from the root of *that* directory. 

---

## Evaluation Metrics and Visualization


---

## UI and Demo

A **real-time interactive demo** showcases valence–arousal estimation directly from webcam input. You can check a showcase video here: [demo](https://www.youtube.com/watch?v=waxnplqpPgs)

**For demo details**, including setup instructions and usage guide, please refer to:  [`demo/README.md`](./demo/README.md)



## ⚙️ Requirements

A detailed list of dependencies and installation instructions will be included in each submodule’s README.


---

## License & Citations

If you use this code in a publication or project, please cite the original EmoNet paper and follow all related license terms.

> **“Estimation of continuous valence and arousal levels from faces in naturalistic conditions”**
> Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos, and Maja Pantic,
> *Nature Machine Intelligence*, January 2021.
