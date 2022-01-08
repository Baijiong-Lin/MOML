# MOML

This repository contains the source code of our paper "Multi-Objective Meta Learning" (NeurIPS 2021).



## Environment

- Python 3.7.10
- torch 1.8.0+cu111
- torchvision 0.9.0+cu111

## Dataset

- **Office-31**: Download from https://www.cc.gatech.edu/~jhoffman/domainadapt/#datasets_code
- **Office-Home**: Download from https://www.hemanthdv.org/officeHomeDataset.html
- **NYUv2**: Download the pre-processed data from https://github.com/lorenmt/mtan#image-to-image-predictions-one-to-many

## Experiments

### MTL

1. Training on the NYUv2 dataset with the MOML method, you can run the code below (default option is training without data augmentation)

```shell
cd ./MTL
python moml_nyu.py --gpu_id [GPU_ID] --model [DMTL, MTAN] --MGDA --dataset_path [ROOT]
```

2. Training on the Office-31 or Office-Home dataset with the MOML method, you can run the code below

```shell
cd ./MTL
python moml_office.py --gpu_id [GPU_ID] --model [DMTL, MTAN] --dataset [office-31, office-home] --batchsize 64 --MGDA --dataroot [ROOT]
```

### SSDA

Training on the Office-31 dataset with the MOML+MME method, you can run the code below

```shell
cd ./SSDA
python moml_MME.py --gpu_id [GPU_ID] --source [SOURCE] --taeget [TARGET] --MGDA
```

## Citation

If you found this code/work to be useful in your own research, please considering citing the following:

```latex
@inproceedings{ye2021moml,
  title={Multi-Objective Meta Learning},
  author={Ye, Feiyang and Lin, Baijiong and Yue, Zhixiong and Guo, Pengxin and Xiao, Qiao and Zhang, Yu},
  booktitle={Proceedings of the 35th Annual Conference on Neural Information Processing Systems},
  year={2021}
}
```

## Acknowledgement

Thanks for the public code base https://github.com/lorenmt/mtan, https://github.com/VisionLearningGroup/SSDA_MME, and https://github.com/isl-org/MultiObjectiveOptimization.

## Contact

If you have any questions, please contact `linbj@mail.sustech.edu.cn`.
