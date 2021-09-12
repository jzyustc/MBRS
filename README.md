# MBRS 



## MBRS: Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression



Zhaoyang  Jia, Han Fang, Weiming Zhang (from University of Science and Technology of China)

[[arXiv]](https://arxiv.org/abs/2108.08211) [[PDF]](https://arxiv.org/pdf/2108.08211) 



> This is the source code of paper *MBRS : Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression*, which is received by ACM MM' 21 (oral). Please contact me in *issue* page or email jzy_ustc@mail.ustc.edu.cn if you find bugs. Thanks!



### Requirements

We used these packages/versions in the development of this project.

- Pytorch `1.5.0`
- torchvision `0.3.0a0+ec20315`
- kornia `0.3.0`
- numpy `1.16.4`
- Pillow `6.0.0`
- scipy `1.3.0`



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

