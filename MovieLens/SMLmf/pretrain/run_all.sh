#!/bin/bash
conda activate asmg
python ./train_ml.py --seed 1
python ./train_ml.py --seed 2
python ./train_ml.py --seed 3
python ./train_ml.py --seed 4
python ./train_ml.py --seed 5
