# Mitigating Sample Selection Bias with Robust Domain Adaption in Multimedia Recommendation

## 1. Introduction
This is the implementation code of DAMCAR, a novel debiasing framework that introduces **D**omain **A**daptation to mitigate Sample Selection Bias (SSB) 
in **M**ultimedia **CA**scade **R**ecommendation systems.

## 2. Preparation
### 2.1 Requirements
- cuda 11.7
- python 3.8.0
- pytorch 2.0.1
- numpy 1.24.3
- pandas 2.0.3
- scikit-learn 1.3.2
- tqdm 4.66.2
- faiss-gpu 1.7.1

### 2.2 Feature Processing & Data Preparation
To process the multi-modal features, please use: 
```
python ./preprocessing/process_mm_features_[dataset_name].py
```
After processing, execute the data preparation procedure for model training as follows:
```
python ./preprocessing/prepare_data_[dataset_name].py
```
**NOTE**: Please modify the path in config.py to the folder where you store the data.

### 2.3. Ranking Model Pre-Training
To pre-train the ranking model, run the following command:
```
python train_ranking_model.py -d [dataset_name] -r [ranking_model_name]
```

## 3. DAMCAR Training and Evaluation
To execute DAMCAR training and evaluation, please use
```
python train_DAMCAR.py -d [dataset_name] -m [debias_method] -r [ranking_model_name]
```

## 4. Acknowledgments
Our code is based on the implementation of [DeepMatch](https://github.com/bbruceyuan/DeepMatch-Torch) and [DeepCTR](https://github.com/shenweichen/DeepCTR-Torch).
