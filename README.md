# Network Calibration
This repository is for evaluating calibration of neural networks and training the neural network with various calibration method.

Currently supported pre-hoc methods are:
- Cross-entropy (baseline)
- Focal Loss [Calibrating Deep Neural Networks using Focal Loss](https://proceedings.neurips.cc/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf)
- Mixup [On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks](https://arxiv.org/abs/1905.11001)
- OECC [Outlier Exposure with Confidence Control for Out-of-Distribution Detection](https://arxiv.org/abs/1906.03509)
- OE [Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/abs/1812.04606)
- Soft-Calibration [Soft Calibration Objectives for Neural Networks](https://arxiv.org/abs/2108.00106)

Currently supported post-hoc methods are:
- Temperature Scaling
- Vector Scaling
- Matrix Scaling
- Splines Fitting [Calibration of Neural Networks using Splines](https://arxiv.org/abs/2006.12800)

## Updates & TODO Lists
- [ ] Automated norm correction function for various architectures.

## Getting Started

### Environment Setup

Tested on GTX2080TI with python 3.7, pytorch 1.13.0, torchvision 0.14.0, CUDA 11.1.

1. Install dependencies
- [timm](https://timm.fast.ai/)

2. Set up a python environment
```
conda create -n test_env python=3.8
conda activate test_env
pip install torch torchvision
pip install timm
```

## Train & Evaluation

### Trainining of the model


```
python train_net.py --epoch 100
```

### Evaluation on test dataset
```
python test_net.py --vis_results
```

## License

The source code of this repository is released only for academic use. See the [license](./LICENSE.md) file for details.


## License
Distributed under the MIT License.

## Acknowledgments
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2022-0-00951, Development of Uncertainty-Aware Agents Learning by Asking Questions)
