# MBRS 



## MBRS: Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression



Zhaoyang  Jia, Han Fang, Weiming Zhang (from University of Science and Technology of China)

[[arXiv]](https://arxiv.org/abs/2108.08211) [[PDF]](https://arxiv.org/pdf/2108.08211) 



> This is the source code of paper *MBRS : Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression*, which is received by ACM MM' 21 (oral). Please contact me in *issue* page or email jzy_ustc@mail.ustc.edu.cn if you find bugs. Thanks!


****
### Updated 10/03/2021 : training of diffusion model

Because of the training process for models with diffusion model (details about diffusion model is in the [paper](https://arxiv.org/pdf/2108.08211)) is not stable, we update the training process for a more stable process. 

- **Raw training** is for 128x128 images and 30 bits message, and it's 256 dimensions after the Fully Connection embedding. Batch size is 16, and learning rate is 1e-3. We train it for 300 epochs and apply early stopping at 110 epoch, decided by the validation results. In this way we get the pretrained model and the result in the [paper](https://arxiv.org/pdf/2108.08211) is based on it. 

- However, this training process is not stable for crop robustness ([Crop attack · Issue #2](https://github.com/jzyustc/MBRS/issues/2)), that is, the validation result for Crop varies from BER=2% to BER=25% for *Crop(p=3.5%)*, and it's hard to guarantee the as good results as we get. 

- To solve it, we **update the training process** in an easy but useful way. 
  - First we train the model like in **raw training** for 100 epochs and apply early stop at 92 epoch to get a suboptimum model (BER = 20% and PSNR=29.75 for *Crop(p=3.5%)*).
  - Then we finetune the model for 50 epochs with the same settings but learning rate=1e-4, and apply early stop at 13 epoch to get the optimum model (BER = 1.85% and PSNR=30.89 for *Crop(p=3.5%)*). 

Wish this can help you :) 


****

### Requirements

We used these packages/versions in the development of this project.

- Pytorch `1.5.0`
- torchvision `0.3.0a0+ec20315`
- kornia `0.3.0`
- numpy `1.16.4`
- Pillow `6.0.0`
- scipy `1.3.0`


****

### Dataset prepare

Please download ImageNet or COCO datasets, and push them into `datasets` folder like this : 

```
├── datasets
│   ├── train
│   │   ├── xxx.jpg
│   │   ├── ...
│   ├── test
│   │   ├── xxx.jpg
│   │   ├── ...
│   ├── validation
│   │   ├── xxx.jpg
│   │   ├── ...
├── ...
├── results
```

For more details about the used datasets, please  read the original [paper](https://arxiv.org/pdf/2108.08211).



### Pretrained Models

Please download pretrained models in [Google Drive](https://drive.google.com/drive/folders/1A_SAqvU2vMsHxki0s9m9rKa-g8B6aghe?usp=sharing) and put the in path `results/xxx/models/`. (xxx is the name of the project, e.g. MBRS_256_m256)


****


### Train

Change the settings in json file `train_settings.json`, then run :

```bash
python train.py
```

The logging file and results will be saved at `results/xxx/`



### Test

Change the settings in json file `test_settings.json`, then run :

```bash
python test.py
```

The logging file and results will be saved at `results/xxx/`


****


### Citation

Please cite our paper if you find this repo useful!

```
@inproceedings{jia2021mbrs,
  title={MBRS: Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression},
  author={Zhaoyang  Jia, Han Fang and Weiming Zhang},
  booktitle={arXiv:2108.08211},
  year={2021}
}
```





Contact: [jzy_ustc@mail.ustc.edu.cn](mailto:jzy_ustc@mail.ustc.edu.cn)

