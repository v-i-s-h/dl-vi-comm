# End to End Communication System design using Deep Learning
This repository is the code for the paper *Raj, V. and Kalyani, S., 2020. Design of communication systems using deep learning: A variational inference perspective. IEEE Transactions on Cognitive Communications and Networking*. 

**Abstract**

Recent research in the design of end to end communication system using deep learning has produced models which can outperform traditional communication schemes. Most of these architectures leveraged autoencoders to design the encoder at the transmitter and decoder at the receiver and train them jointly by modeling transmit symbols as latent codes from the encoder. However, in communication systems, the receiver has to work with noise corrupted versions of transmit symbols. Traditional autoencoders are not designed to work with latent codes corrupted with noise. In this work, we provide a framework to design end to end communication systems which accounts for the existence of noise corrupted transmit symbols. The proposed method uses deep neural architecture. An objective function for optimizing these models is derived based on the concepts of variational inference. Further, domain knowledge such as channel type can be systematically integrated into the objective. Through numerical simulation, the proposed method is shown to consistently produce models with better packing density and achieving it faster in multiple popular channel models as compared to the previous works leveraging deep learning models. 

| [arXiv](https://arxiv.org/abs/1904.08559) | [IEEE](https://ieeexplore.ieee.org/abstract/document/9056790/) | [Scholar](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=3364499034871405494&as_sdt=5) |

## How to use?
All simulations are tested in TF-1.12. Simulations for each channel is organized under individual folders.
0. Use `environment.yml` to set up a conda environment.
1. Run `main.py` with appropriate options. Refer `driver.sh` for examples.
2. Run `analysis_energy.ipynb` and get `*_summary.h5` files from `*_summary.dil`. This will create summary files with packing density information.
3. Run `analysis_figures.ipynb` to plot the figures. Input should be `*_summary.h5`.

#### Cite as
```
@article{raj2020design,
  title={Design of communication systems using deep learning: A variational inference perspective},
  author={Raj, Vishnu and Kalyani, Sheetal},
  journal={IEEE Transactions on Cognitive Communications and Networking},
  year={2020},
  publisher={IEEE}
}
```

#### Contact Information
For more information, contact first author (VIshnu Raj) at the email address given in the paper.