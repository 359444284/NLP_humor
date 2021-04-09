# Team：CHAOYUDENG in HaHackathon: Detecting and Rating Humor and Offense


## Environment Configuration:
Install python 3.6
https://www.python.org/downloads/release/python-368/

- Pytorch: pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0-fhttps://download.pytorch.org/whl/torch_stable.html
- transformers:  pip install transformers==3.5.0
- sentencepiece:  pip install sentencepiece==0.1.91
- numpy:  pip install numpy==1.19.4
- matplotlib:  pip install matplotlib==3.3.3
- scikit-learn:  pip install scikit-learn==0.23.2
- pandas:  pip install pandas==1.1.4
- Pillow： pip install Pillow==8.0.1

## Dataset:
- train data https://competitions.codalab.org/competitions/27446#participate-get_data (./datas/task1/train/)
- public data (test data) https://competitions.codalab.org/competitions/27446#participate-get_starting_kit (./public_test.csv)

## Models Used in Essay

Best Model: RoBERTa + MTL + All Layer (loss weighting: 0.4 0 0 0.6)

        python model.py

Other Model:
1. Albert

        python model.py --uncertainty False --all_layer False --weights 1 0 0 --dropout 0.8 0.8 0.8 0.8 --model albert-xxlarge-v2

2. Albert + MTL

        python model.py --uncertainty False --all_layer False --weights 0.85 0.075 0 --dropout 0.8 0.8 0.8 0.8 --model albert-xxlarge-v2
4. Albert + MTL + All Layer

        python model.py --uncertainty False --weights 0.85 0.075 0 --dropout 0.8 0.8 0.8 0.8 --model albert-xxlarge-v2
6. RoBERTa

        python model.py --uncertainty False --all_layer False --weights 1 0 0 --dropout 0.6 0 0 0 
8. RoBERTa + MTL

        python model.py --uncertainty False

## Hyper Parameters And Defluat Value
You can add --+[Parameters] + value to set up you own Hyper Parameters.
- batch_size 8
- epochs 15
- lr 2e-06
- seed 70
- cuda [0, 1, 2]
- uncertainty True
- all_layer True
- weights [0.4, 0, 0]
- model roberta-large
- dropout [0.3, 0.3, 0.3, 0.3]




## References

- https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
- https://zhuanlan.zhihu.com/p/83609874

