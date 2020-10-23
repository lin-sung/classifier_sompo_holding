# Classififier for Sompo Holding POC

The repository contains the codes to train the three class classification model in Sompo Holding POC.

## Codes Structure
```
classifier
├── checkpoints (store checkpoints)
├── data_loader
│   ├── __init__.py
│   ├── data.py
│   └── data_utils.py
├── log
├── models
│   ├── __init__.py
│   ├── base_model.py
│   └── gcn.py
├── utils
│   ├── avgmeter.py
│   ├── graph.py
│   ├── kv2.py
│   ├── torch_utils.py
│   └── utils.py
├── README.md
├── __init__.py
├── model.py (model interface)
├── requirements.txt
├── run_classifier.py (code's entry)
├── split_train_valid.py
├── train.sh (train script for local machine)
├── train_sagemaker.py (train script for sagemaker)
└── trainer.py       
```

## Training

### Split the data
```
python split_train_valid.py [DATA_PATH]
```
It will generate `train.lst` and `val.lst`, which indicate the filenames of training and validation data.

### Training
```
bash train.sh
or 
python train_sagemaker.py
```
At the first time training, please set `preprocessing=True` to process the data. Afterwards, you can set it to `False` to save some time.
### Arguments
```diff
- test: whether to use testing mode, or using train mode (default=False)
- sagemaker: whether use in sagemaker (default=False)
- train_data_dir: the directory of the train data"
- test_data_dir: the directory of the test data
- checkpoints_folder: the folder to save checkpoints
- epochs: maximum epochs
- backbone: model backbone (default: Attention_GCN_backbone)
- checkpoints_dir
- lr
- weight_decay
- cuda: whether to use gpu
- linear_decay: whether to use linear decay in optimizer
- lambda_kv: weight for the kv's loss
- preprocessing: whether preprossess the data (default=False)
- extend_dataset: the directory to extend the data (usually is the ocr output)
```

## Classes
```
"others": 0,
"not be admitted": 1,
"be admitted": 2
```

## Model Zoo
All the models are graph-based, and the model will predict some kv classes (`date` and `patient_status`) to help the classification output. <br/>
1. `Attention_GCN_backbone` <br/>
Transform the output kv logits of textlines as the attention weights, and weighted-sum the graph feature to make prediction. The weights of the two classes (`date` and `patient_status`) are manually designed. <br/>
2. `Jeff_Attention_GCN_backbone` <br/>
Directly predict the classification output without interact with kv outputs. <br/>
3. `SelfAttention_GCN_backbone` <br/>
Similar as 1., but the weights of the two classes (`date` and `patient_status`) are learned by linear function. <br/>
4. `SelfAttention_GCN_classifier` <br/>
Similar as 1., but the attention weights of textlines are directly learned by a linear function.
5. `MultiAttention_GCN_backbone` <br/>
Combine 1. and 4.


