# Network Calibration


- This is blank template to get started.
- You can edit this template to suit your needs.
- It is recommended to include an image in the readme.
- You can find more awesome README templates from [google](https://www.google.com/search?q=awesome+readme+templates&oq=awe&aqs=chrome.0.69i59j69i57j69i61.1243j0j4&sourceid=chrome&ie=UTF-8)

<img src="./figures/screenshot.png" height="600">


## Updates & TODO Lists
- [X] (2022.12.22) README_TEMPLATES.md is released.
- [ ] Adding more samples.


## Getting Started

### Environment Setup

Tested on Titan RTX with python 3.7, pytorch 1.8.0, torchvision 0.9.0, CUDA 10.2 / 11.1 and detectron2 v0.5 / v0.6

1. Install dependencies
```
sudo apt update && sudo apt upgrade
```

2. Set up a python environment
```
conda create -n test_env python=3.8
conda activate test_env
pip install torch torchvision
python setup.py build develop
```

## Train & Evaluation

### Dataset Preparation
1. Download `sample dataset' from MAT.
```
wget sample_dataset.com
```

2. Extract it to `sample folder`
```
tar -xvf sample_dataset.tar
```

3. Organize the folders as follows
```
test
├── output
└── datasets
       └── sample_dataset
              └──annotations
              └──train
              └──val       
```
### Train on sample dataset
```
python train_net.py --epoch 100
```

### Evaluation on test dataset
```
python test_net.py --vis_results
```

## License

The source code of this repository is released only for academic use. See the [license](./LICENSE.md) file for details.

## Notes

Some codes are rewritten from
- [Best-README-Template](https://github.com/othneildrew/Best-README-Template/edit/master/BLANK_README.md)


## Authors
- **Seunghyeok Back** [seungback](https://github.com/SeungBack)

## License
Distributed under the MIT License.

## Acknowledgments
This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by Korea goverment(MSIT) (No.2019-0-01335, Development of AI technology to generate and validate the task plan for assembling furniture in the real and virtual environment by understanding the unstructured multi-modal information from the assembly manual.
