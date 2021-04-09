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

## Models

BEST: RoBERTa + MTL + All Layer
        python 

1. Albert


   
2. Albert + MTL
3. Albert + MTL + All Layer
4. RoBERTa
5. RoBERTa + MTL




## References

- https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
- https://zhuanlan.zhihu.com/p/83609874

