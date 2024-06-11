# PFR
This repo is the implementation of the following paper:

**Pseudo-Set Frequency Refinement Architecture for Fine-Grained Few-Shot Class-Incremental Learning** (PR 2024)

## Abstract
Few-shot class-incremental learning was introduced to solve the model adaptation problem for new incremental classes with only a few examples while still remaining effective for old data. Although recent state-of-the-art methods make some progress in improving system robustness on common datasets, they fail to work on fine-grained datasets where inter-class differences are small. The problem is mainly caused by: 1) the overlapping of new data and old data in the feature space during incremental learning, which means old samples can be falsely classified as newly introduced classes and induce catastrophic forgetting phenomena; 2) lacking discriminative feature learning ability to identify fine-grained objects. In this paper, a novel Pseudo-set Frequency Refinement (PFR) architecture is proposed to tackle these problems. We design a pseudo-set training strategy to mimic the incremental learning scenarios so that the model can better adapt to novel data in future incremental sessions. Furthermore, separate adaptation tasks are developed by utilizing frequency-based information to refine the original features and address the above challenging problems. More specifically, the high and low-frequency components of the images are employed to enrich the discriminative feature analysis ability and incremental learning ability of the model respectively. The refined features are used to perform inter-class and inter-set analyses. Extensive experiments show that the proposed method consistently outperforms the state-of-the-art methods on four fine-grained datasets.

## Training Scripts

- Train CUB200

    ```
    python train.py -project pfr -dataset cub200 -gamma 0.25 -lr_base 0.002 -decay 0.0005 -epochs_base 120 -schedule Milestone -milestones 70 90 110 -gpu '0,1,2,3' -temperature 16 -dataroot YOUR_DATA_ROOT -alpha 0.03
    ```
- Train StanfordDog
    ```
    python train.py -projec pfr -dataset StanfordDog -gamma 0.1 -lr_base 0.2 -decay 0.0005 -epochs_base 400 -schedule Cosine -gpu '0,1,2,3' -temperature 16 -dataroot YOUR_DATA_ROOT -alpha 0.3
    ```
- Train Aircraft
    ```
    python train.py -projec pfr -dataset Aircraft -gamma 0.1 -lr_base 0.2 -decay 0.0005 -epochs_base 400 -schedule Cosine -gpu '0,1,2,3' -temperature 16 -alpha 0.3 -dataroot YOUR_DATA_ROOT
    ```
- Train StanfordCar
    ```
    python train.py -projec pfr -dataset StanfordCar -gamma 0.1 -lr_base 0.2 -decay 0.0005 -epochs_base 400 -schedule Cosine -gpu '0,1,2,3' -temperature 16 -dataroot YOUR_DATA_ROOT -alpha 0.3
    ```


## Acknowledgment
We thank the following repos for providing helpful components/functions in our work.

- [CEC](https://github.com/icoz69/CEC-CVPR2021)

- [FACT](https://github.com/zhoudw-zdw/CVPR22-Fact)

- [TEEN](https://github.com/wangkiw/TEEN)
