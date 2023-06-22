# Frequency & Channel Attention for Computationally Efficient Sound Event Detection

Official implementation of <br>
 - **Frequency & Channel Attention for Computationally Efficient Sound Event Detection** (Submitted to DCASE 2023 workshop) <br>
by Hyeonuk Nam, Seong-Hu Kim, Doekki Min, Yong-Hwa Park <br>[![arXiv](https://img.shields.io/badge/arXiv-2306.11277-brightgreen)](https://arxiv.org/abs/2306.11277)<br>


## Requirements
Python version of 3.7.10 is used with following libraries
- pytorch==1.8.0
- pytorch-lightning==1.2.4
- pytorchaudio==0.8.0
- scipy==1.4.1
- pandas==1.1.3
- numpy==1.19.2


other requrements in [requirements.txt](./requirements.txt)


## Datasets
You can download datasets by reffering to [DCASE 2021 Task 4 description page](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments) or [DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task). Then, set the dataset directories in [config yaml files](./configs/) accordingly. You need DESED real datasets (weak/unlabeled in domain/validation/public eval) and DESED synthetic datasets (train/validation).

## Training
You can train and save model in `exps` folder by running:
```shell
python main.py
```
default model in the config.yaml is SE+tfwSE

## Test with saved models
You can test saved models by running:
```shell
python main.py -s saved_models/SE+tfwSE/best
```
this example tests the best SE+tfwSE model saved.



## Reference
- [DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task) <br>
- [Sound event detection with FilterAugment](https://github.com/frednam93/FilterAugSED) <br>
- [Temporal Dynamic CNN for text-independent speaker verification](https://https://github.com/shkim816/temporal_dynamic_cnn)
- [Frequency Dynamic Convolution-Recurrent Neural Network (FDY-CRNN) for Sound Event Detection](https://github.com/frednam93/FDY-SED)

## Citation & Contact
If this repository helped your works, please cite papers below! 3rd paper is about data augmentation method called FilterAugment which is applied to this work.
```bib
@article{nam2023frequency,
      title={Frequency & Channel Attention for Computationally Efficient Sound Event Detection}, 
      author={Hyeonuk Nam and Seong-Hu Kim and Deokki Min and Yong-Hwa Park},
      journal={arXiv preprint arXiv:2306.11277},
      year={2023},
}

@inproceedings{nam22_interspeech,
      author={Hyeonuk Nam and Seong-Hu Kim and Byeong-Yun Ko and Yong-Hwa Park},
      title={{Frequency Dynamic Convolution: Frequency-Adaptive Pattern Recognition for Sound Event Detection}},
      year=2022,
      booktitle={Proc. Interspeech 2022},
      pages={2763--2767},
      doi={10.21437/Interspeech.2022-10127}
}

@INPROCEEDINGS{nam2021filteraugment,
    author={Nam, Hyeonuk and Kim, Seong-Hu and Park, Yong-Hwa},
    booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
    title={Filteraugment: An Acoustic Environmental Data Augmentation Method}, 
    year={2022},
    pages={4308-4312},
    doi={10.1109/ICASSP43922.2022.9747680}
}
```
Please contact Hyeonuk Nam at frednam@kaist.ac.kr for any query.
