# PYGC: a PinYin Language Model Guided Correction Model for Chinese Spell Checking
![image](./architecture.png)

## Requirements
- python 3.8
- transformers 4.12.5
- pytorch-lightning 1.5.1
- pypinyin

## Preparation
### 1. Data
Download the SIGHAN and CSCD-IME dataset from Baidu Netdisk and unzip it as folder "PYGC/data/"：  
Link: https://pan.baidu.com/s/1F6gaQfcglwH5j61t3rN0vw Password：1ne9  
### 2. Pinyin LM checkpoints and Pretrained checkpoints 
Download from Baidu Netdisk and unzip it as folder "PYGC/checkpoints/":  
Link：https://pan.baidu.com/s/15Age7n73En0fWLO6mRw9MA 
Password：bu4r 

## Inference
We provide the SIGHAN inference checkpoint from Baidu Netdisk. Please download it and unzip it as folder "PYGC/outputs/finetuned/":    
Link：https://pan.baidu.com/s/1iXuSi1eIHjGV_UnlV8k4Jg 
Password：ds53 

You can run this file to predict SIGHAN or SIGHAN-Isolation performance:  
`bash predict.sh`

