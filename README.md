# Team：CHAOYU DENG in SemEval 2022 Task 4: Patronizing and Condescending Language Detection
member： CHAOYU DENG， Kang, Yining， Yao, Jinli

## Environment Configuration:
Install python 3.6
https://www.python.org/downloads/release/python-368/

- Pytorch: pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0-fhttps://download.pytorch.org/whl/torch_stable.html
- transformers:  pip install transformers==4.16.2
- sentencepiece:  pip install sentencepiece==0.1.91
- numpy:  pip install numpy==1.19.4
- matplotlib:  pip install matplotlib==3.3.3
- scikit-learn:  pip install scikit-learn==1.0.1
- pandas:  pip install pandas==1.3.4
- Pillow： pip install Pillow==8.0.1
- tqdm: pip install tqdm

## Models Used in Essay

Defult Model: deberta-v3-large + MTD + lstm_gru lr:5e-6, elr:0.9, batch size:4 loss:DI 0.5 3N seed:12 （val: 0.6630 0.6030 0.6316）

        python train_model.py

Other Model:
1. Bert_lastClsSep lr:3e-6 elr 0.95 batch size 4 loss DI 0.3  + MTD + ems + FGM seed:15 （val: 0.6510 0.6020 0.6265）

        python train_model.py --model_output cls_spe --lr 3e-6 --dice_fact 0.3 --sample_num 2.5 --seed 15 --epochs 15 --atk FGM --ema True

2. Bert_lastClsSep lr:5e-6 elr 0.9 batch size 4 loss DI 0.3 + MTD + ems + clamp + R-drop seed:15 (val: 0.5764    0.6633    0.6168)

        python train_model.py --model_output cls_spe --lr 5e-6 --dice_fact 0.3 --sample_num 2.5 --seed 15 --ema True --grad_clamp True --r_drop True

## Hyper Parameters And Defluat Value
You can add --+[Parameters] + value to set up you own Hyper Parameters.
- batch_size 4
- epochs 12
- lr 5e-6
- seed 12
- model microsoft/deberta-v3-large
- model_output lstm_gru
- dice_fact 0.5
- atk ''
- ema False
- grad_clamp False
- r_drop False
- sample_num 3

## Result
python predict.py  (get the submission file and validation result)

