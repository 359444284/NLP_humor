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

## Experimental results (Baseline vs CSD-PLUS)

I. SYN

        1. python3 runner.py -w DataSet/SYN/SYN_W.txt -z DataSet/SYN/SYN_2_Z.txt -s DataSet/SYN/pat_2.txt -k 2 -o 0.5 -c 50 -m 1000 -d 10 -t 3 -e 5
        2. python3 runner.py -w DataSet/SYN/SYN_W.txt -z DataSet/SYN/SYN_3_Z.txt -s DataSet/SYN/pat_3.txt -k 3 -o 0.5 -c 50 -m 1000 -d 10 -t 3 -e 5
        3. python3 runner.py -w DataSet/SYN/SYN_W.txt -z DataSet/SYN/SYN_4_Z.txt -s DataSet/SYN/pat_4.txt -k 4 -o 0.5 -c 50 -m 2000 -d 10 -t 3 -e 5
        4. python3 runner.py -w DataSet/SYN/SYN_W.txt -z DataSet/SYN/SYN_5_Z.txt -s DataSet/SYN/pat_5.txt -k 5 -o 0.5 -c 25 -m 3000 -d 10 -t 3 -e 5
        5. python3 runner.py -w DataSet/SYN/SYN_W.txt -z DataSet/SYN/S5_Z.txt  -s DataSet/SYN/S5.txt  -k 3 -o 0.5 -c 50 -m 2000 -d 10 -t 3 -e 5
        6. python3 runner.py -w DataSet/SYN/SYN_W.txt -z DataSet/SYN/S10_Z.txt -s DataSet/SYN/S10.txt -k 3 -o 0.5 -c 50 -m 1000 -d 10 -t 3 -e 5
        7. python3 runner.py -w DataSet/SYN/SYN_W.txt -z DataSet/SYN/S15_Z.txt -s DataSet/SYN/S15.txt -k 3 -o 0.5 -c 50 -m 3000 -d 10 -t 3 -e 5
        8. python3 runner.py -w DataSet/SYN/SYN_W.txt -z DataSet/SYN/S20_Z.txt -s DataSet/SYN/S20.txt -k 3 -o 0.5 -c 50 -m 3000 -d 10 -t 3 -e 5
   
II. TRU


        1. python3 runner.py -w DataSet/TRU/TRU_W.txt -z DataSet/TRU/TRU_3_Z.txt -s DataSet/TRU/pat_3.txt -k 3 -o 0.5 -c 50 -m 1000 -d 10 -t 10 -e 5
        2. python3 runner.py -w DataSet/TRU/TRU_W.txt -z DataSet/TRU/TRU_4_Z.txt -s DataSet/TRU/pat_4.txt -k 4 -o 0.5 -c 50 -m 1000 -d 10 -t 10 -e 5
        3. python3 runner.py -w DataSet/TRU/TRU_W.txt -z DataSet/TRU/TRU_5_Z.txt -s DataSet/TRU/pat_5.txt -k 5 -o 0.5 -c 50 -m 1000 -d 10 -t 10 -e 5
        4. python3 runner.py -w DataSet/TRU/TRU_W.txt -z DataSet/TRU/TRU_6_Z.txt -s DataSet/TRU/pat_6.txt -k 6 -o 0.5 -c 50 -m 1000 -d 10 -t 10 -e 5
        5. python3 runner.py -w DataSet/TRU/TRU_W.txt -z DataSet/TRU/S20_Z.txt -s DataSet/TRU/S20.txt -k 4 -o 0.5 -c 25 -m 1000 -d 10 -t 5 -e 5
        6. python3 runner.py -w DataSet/TRU/TRU_W.txt -z DataSet/TRU/S40_Z.txt -s DataSet/TRU/S40.txt -k 4 -o 0.5 -c 25 -m 1000 -d 10 -t 5 -e 5
        7. python3 runner.py -w DataSet/TRU/TRU_W.txt -z DataSet/TRU/S60_Z.txt -s DataSet/TRU/S60.txt -k 4 -o 0.5 -c 25 -m 1000 -d 10 -t 5 -e 5
        8. python3 runner.py -w DataSet/TRU/TRU_W.txt -z DataSet/TRU/S80_Z.txt -s DataSet/TRU/S80.txt -k 4 -o 0.5 -c 25 -m 1000 -d 10 -t 5 -e 5

III. DNA

        1. python3 runner.py -w DataSet/DNA/DNA_W.txt -z DataSet/DNA/DNA_2_Z.txt -s DataSet/DNA/pat_2.txt -k 2 -o 0.5 -c 50 -m 1000 -d 10 -t 10 -e 5
        2. python3 runner.py -w DataSet/DNA/DNA_W.txt -z DataSet/DNA/DNA_3_Z.txt -s DataSet/DNA/pat_3.txt -k 3 -o 0.5 -c 50 -m 1000 -d 10 -t 10 -e 5
        3. python3 runner.py -w DataSet/DNA/DNA_W.txt -z DataSet/DNA/DNA_4_Z.txt -s DataSet/DNA/pat_4.txt -k 4 -o 0.5 -c 50 -m 2000 -d 10 -t 10 -e 5
        4. python3 runner.py -w DataSet/DNA/DNA_W.txt -z DataSet/DNA/DNA_5_Z.txt -s DataSet/DNA/pat_5.txt -k 5 -o 0.5 -c 50 -m 3000 -d 10 -t 10 -e 5
        5. python3 runner.py -w DataSet/DNA/DNA_W.txt -z DataSet/DNA/S20_Z.txt  -s DataSet/DNA/S20.txt -k 4 -o 0.5 -c 50 -m 1000 -d 10 -t 10 -e 5
        6. python3 runner.py -w DataSet/DNA/DNA_W.txt -z DataSet/DNA/S40_Z.txt -s DataSet/DNA/S40.txt -k 4 -o 0.5 -c 50 -m 1000 -d 10 -t 10 -e 5
        7. python3 runner.py -w DataSet/DNA/DNA_W.txt -z DataSet/DNA/S60_Z.txt -s DataSet/DNA/S60.txt -k 4 -o 0.5 -c 50 -m 1000 -d 10 -t 10 -e 5
        8. python3 runner.py -w DataSet/DNA/DNA_W.txt -z DataSet/DNA/S80_Z.txt -s DataSet/DNA/S80.txt -k 4 -o 0.5 -c 50 -m 1000 -d 10 -t 10 -e 5



## References

- https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
- Bernardini, Giulia, et al. "Hide and Mine in Strings: Hardness and Algorithms." International Conference on Data Mining (ICDM). 2020.
- Browne, C. B., Powley, E., Whitehouse, D., Lucas, S. M., Cowling, P. I., Rohlfshagen, P., ... & Colton, S. (2012). A survey of monte carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in games, 4(1), 1-43.

