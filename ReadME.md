# HCL_Text2AMR
Source code of ACL2022 short paper: [Hierarchical Curriculum Learning for AMR Parsing](https://arxiv.org/abs/2110.07855)

# 🔥 Introduction
The sequence-to-sequence models have become a main stream paradigm for AMR parsing. However, there exists a gap between their flat training objective (i.e., equally treats all output tokens) and the hierarchical AMR structure, which limits the ability of model to learn the inherent hierarchical structure of AMR. To bridge this gap, we propose a Hierarchical Curriculum Learning (HCL) framework with Structure-level (SC) and Instance-level Curricula (IC). SC switches progressively from core to detail AMR semantic elements while IC transits from structure-simple to -complex AMR instances during training.

![overview](./fig/overview.pdf)



# 🚀 How to use our code?
## 💾 Enviroment
```
pip install -r requirements.txt
pip install -e .
```
## 🏋🏻‍♂️ Train the model
Modify the data path in `configs/HCL.yaml`. The run:
```
bash bash/train.sh [gpu id]
```


## 🥷 Evaluation
### inference test data
```
bash bash/predict.sh [gpu_id] [AMR dataset version (e.g., 2)] [checpoint path] 
```

### post processing
use [BLINK](https://github.com/facebookresearch/BLINK) to post process the output of the model
```
bash bash/blinkified.sh [gpu_id] [AMR dataset version]
```

### fine grained evaluation
```
bash bash/fine-eval.sh [AMR dataset version]
```

# 🌝 Citation
If you use our code, plearse cite our paper:
```
@article{wang2021hierarchical,
  title={Hierarchical Curriculum Learning for AMR Parsing},
  author={Wang, Peiyi and Chen, Liang and Liu, Tianyu and Chang, Baobao and Sui, Zhifang},
  journal={arXiv preprint arXiv:2110.07855},
  year={2021}
}
```

# 🌝 Acknowledgements
Our code is based on [SPRING](https://github.com/SapienzaNLP/spring). Thanks for their high quality open codebase.  

# Pretrain Models
- Todo
