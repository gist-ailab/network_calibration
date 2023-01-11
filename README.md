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
conda create -n test_env python=3.7
conda activate test_env
pip install torch torchvision
pip install timm
```

## Train & Evaluation

### Trainining of the model
```
# Example of setting the configure files
|-conf
||-aircraft.json
||-cifar10.json
|-train.py
|-...

# aircraft.conf
{
    "epoch" : "100",                                  # Number of epochs for training
    "id_dataset" : "/SSDe/yyg/FGVC-Aircraft",         # Datset path
    "batch_size" : 32,                                # batch size
    "save_path" : "/SSDe/yyg/calibration/Aircrafts/", # Save path
    "num_classes" : 100                               # Number of classes
}
```

```
# train the ResNet50 model for FGVC-aircrfat with cross-entropy loss on GPU #0
python train.py --inlier-data aircraft --method ce --net resnet50 --gpu 0 --save_path baseline
```

### Evaluation on test dataset
```
# Evalation of ResNet50 model for FGVC-aircraft with temperature scaling method on GPU #0
python eval_fg.py --inlier-data aircraft --method temperature --save_path baseline --gpu 0
```

## License

The source code of this repository is released only for academic use. See the [license](./LICENSE.md) file for details.


## License
Distributed under the MIT License.

## Acknowledgments
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2022-0-00951, Development of Uncertainty-Aware Agents Learning by Asking Questions)
