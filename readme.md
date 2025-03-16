<h1 align="center"> No Pains, More Gains: Recycling Sub-Salient Patches for Efficient High-Resolution Image Recognition </h1>

<p align="center"> CVPR 2025 </p>

## Abstract
<img src="./Figure/figure1.png" width="50%" align="right" />
Over the last decade, many notable methods have emerged to tackle the computational resource challenge of the high resolution image recognition (HRIR). They typically focus on identifying and aggregating a few salient regions for classification, discarding sub-salient areas for low training consumption. Nevertheless, many HRIR tasks necessitate the exploration of wider regions to model objects and contexts, which limits their performance in such scenarios. To address this issue, we present a DBPS strategy to enable training with more patches at low consumption. Specifically, in addition to a fundamental buffer that stores the embeddings of most salient patches, DBPS further employs an auxiliary buffer to recycle those sub-salient ones. To reduce the computational cost associated with gradients of sub-salient patches, these patches are primarily used in the forward pass to provide sufficient information for classification. Meanwhile, only the gradients of the salient patches are back-propagated to update the entire network. Moreover, we design a Multiple Instance Learning (MIL) architecture that leverages aggregated information from salient patches to filter out uninformative background within sub-salient patches for better accuracy. Besides, we introduce the random patch drop to accelerate training process and uncover informative regions. Experiment results demonstrate the superiority of our method in terms of both accuracy and training consumption against other advanced methods.

## DBPS Pipeline
<p align="center">
  <img width="95%" src="./Figure/figure2.png">
</p>

## 1. Requirements

pytorch=1.13.0,torchvision=0.14.0



## 2. Training & Testing

- Train the model on CAMELYON16 dataset:

    `python c16_main1.py`
 
- Train the model on the other five datasets:

    `python main1.py`/`python main2.py`

- Set experiment settings:
     
    browse `config` folder

    # Acknowledgement
This code is borrowed from [[IPS-Transformer](https://github.com/benbergner/ips)] If you use the part of code, you should cite both our work and IPS-Transformer:
```bibtex
```bibtex
@inproceedings{bergner2022iterative,
  title={Iterative patch selection for high-resolution image recognition},
  author={Bergner, Benjamin and Lippert, Christoph and Mahendran, Aravindh},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
```  

    
