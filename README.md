# DDSA using Synthetic Training Data

Digital subtraction angiography (DSA) is a fluoroscopy method primarily used for the diagnosis of cardiovascular diseases.
Deep learning-based DSA (DDSA) is developed to extract DSA-like images directly from fluoroscopic images, which helps in saving dose while improving image quality.
In this work, we developed a method generating synthetic DSA image pairs to train neural network models for DSA image extraction with CT images and simulated vessels.
The synthetic DSA targets are free of typical artifacts and noise commonly found in conventional DSA targets for DDSA model training.
Benefiting from diverse synthetic training data and accurate synthetic DSA targets, models trained on the synthetic data outperform models trained on clinical data in both visual and quantitative assessments.
This approach compensates for the paucity and inadequacy of clinical DSA data.

For more information about this work, please read our [Medical Physics 2024 paper](http://doi.org/10.1002/mp.16973)

> Duan, L., Eulig, E., Knaup, M., Adamus, R., Lell, M., & Kachelrieß, M. "Training of a Deep Learning Based Digital Subtraction Angiography Method using Synthetic Data."


## Requirements
```
python 3.7.1
pytorch 1.8.2
```

## Data

The synthetic vascular projection images we utilized in this work are available for download [here](https://b2share.fz-juelich.de/records/3a652a2089ae4b84bdacc40c676e7825).
To generate vascular projection images:  
```
1. Run SimVessels\stringGenerator.py  
2. Run SimVessels\projectionGenerator.py  
```


## Getting started with training
### Train model using synthetic data
  - Copy your CT projection images into `CT_projs\`
  - Download the [vascular image data](https://b2share.fz-juelich.de/records/3a652a2089ae4b84bdacc40c676e7825) into `vessels\`
  - Run the default training for U-net using `python train_syn.py --cuda --augment --add_noise`. For U-net with adversarial loss, use `python train_GAN.py --cuda --augment --add_noise`.
  - Utilize multiple GPUs for U-net using `python -m torch.distributed.launch --nproc_per_node 4 train_syn.py --m_cuda --devices 0 1 2 3 --augment --add_noise`.

### Train model using clinical data
  - Copy your DSA data into `data/` directory.
  - Modify the corresponding information in the `data/info.xlsx` based on your data.
  - Run the default training for U-net using `python train.py --cuda --augment`. For U-net with adversarial loss, use `python train_GAN_syn.py --cuda --augment`.
  - Utilize multiple GPUs for U-net using `python -m torch.distributed.launch --nproc_per_node 4 train.py --m_cuda --devices 0 1 2 3 --augment`.

### Training Parameters
```
usage: train*.py [-h] [--savepath SAVEPATH] [--datafolder DATAFOLDER] [--cuda] [--m_cuda] [devices] [--nepochs NEPOCHS]
                [--lr LR] [--b1 B1] [--b2 B2] [--criterion CRITERION] [--mbs MBS] [--n_workers N_WORKERS] [--ntrain NTRAIN]
                [--nval NVAL] [--sample_strategy SAMPLE_STRATEGY] [--add_noise] [--augment] [--pBlur PBLUR] [--pAffine PAFFINE]
                [--pMultiply PMULTIPLY] [--pContrast PCONTRAST] [--init_patch INIT_PATCH] [--final_patch FINAL_PATCH]
                [--normalization NORMALIZATION] [--init_type INIT_TYPE] [--upmode UPMODE] [--downmode DOWNMODE] [--batchnorm]
                [--dropout DROPOUT] [--use-trained_model] [--pretrained_model] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --savepath SAVEPATH   Where to store the results.
  --datafolder DATAFOLDER
                        Where the rawdata is located.
  --cuda                Enables cuda
  --m_cuda              Enables multiple GPUs
  --devices             Cuda device to use
  --nepochs NEPOCHS     Number of epochs to train for.
  --lr LR               Learning rate, default=0.0001
  --b1 B1               beta1 parameter of Adam.
  --b2 B2               beta2 parameter of Adam.
  --criterion CRITERION
                        L1 or L2 loss
  --mbs MBS             Mini-batch size
  --n_workers N_WORKERS
                        Number of workers to use for preprocessing.
  --ntrain NTRAIN       Number of training samples
  --nval NVAL           Number of validation samples
  --sample_strategy SAMPLE_STRATEGY
                        Sample strategy for the data. If "dataset", sample uniform w.r.t to datasets. If "uniform" sample uniform
                        over all datasets (not recommended).
  --add_noise           Add poisson noise to CT projections
  --augment             Augment the data.
  --pBlur PBLUR         Probability for blur.
  --pAffine PAFFINE     Probability for affine transformations.
  --pMultiply PMULTIPLY
                        Probability for multiplicative augmentation.
  --pContrast PCONTRAST
                        Probability for contrast augmentations.
  --init_patch INIT_PATCH
                        Size of patch before data augmentation.
  --final_patch FINAL_PATCH
                        Size of patch after data augmentation.
  --normalization NORMALIZATION
                        Data normalization. Global or local (not recommended).
  --init_type INIT_TYPE
                        Weight initialization.
  --upmode UPMODE       Upsample mode
  --downmode DOWNMODE   Downsample mode
  --batchnorm           Use Batch-Normalization
  --dropout DROPOUT     Use dropout with provided probability.
  --use_trained_model   Use a trained model for model initialization.')
  --pretrained_model    Pretrained model name
  --seed SEED           manual seed
```


## Test
Run `test.py` to test the U-net model, and `test_GAN.py` for the U-net GAN model.

## Reference
If you find this project useful for your work, please consider citing our Medical Physics paper
```
@article{duan2024-deepdsa,
  author = {Duan, Lizhen and Eulig, Elias and Knaup, Michael and Adamus, Ralf and Lell, Michael and Kachelrieß, Marc},
  title = {Training of a deep learning based digital subtraction angiography method using synthetic data},
  journal = {Medical Physics},
  volume = {51},
  number = {7},
  pages = {4793-4810},
  doi = {https://doi.org/10.1002/mp.16973},
  url = {https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.16973},
  eprint = {https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.16973},
}
```
