#!/bin/sh

gpu=1

python main.py --data_path ./data/umls-Fed3.pkl --name umls_fed3_transe_collection \
              --setting Collection --mode test --model TransE --gpu $gpu