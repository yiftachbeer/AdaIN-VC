# AdaIN-VC

Implementation of the paper [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742) modified from the official one and heavily based on https://github.com/cyhuang-tw/AdaIN-VC.


Requires at least python 3.6. For other dependencies, see requirements.txt.

## Differences from the official implementation

The main difference from the official implementation is the improvements of audio quality due to the use of a neural vocoder (universal vocoder, whose code was from [yistLin/universal-vocoder](https://github.com/yistLin/universal-vocoder)).

Besides, this implementation supports torch.jit, so the full model can be loaded with simply one line:

```python
model = torch.jit.load(model_path)
```

## Usage
The main contribution of this repo is organizing instructions for preprocessing, training and inference into a [runnable notebook](https://github.com/yiftachbeer/AdaIN-VC/blob/master/notebooks/demo.ipynb). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/yiftachbeer/AdaIN-VC/blob/master/notebooks/demo.ipynb)

## Reference

Please cite the paper if you find AdaIN-VC useful.

```bib
@article{chou2019one,
  title={One-shot voice conversion by separating speaker and content representations with instance normalization},
  author={Chou, Ju-chieh and Yeh, Cheng-chieh and Lee, Hung-yi},
  journal={arXiv preprint arXiv:1904.05742},
  year={2019}
}
```
