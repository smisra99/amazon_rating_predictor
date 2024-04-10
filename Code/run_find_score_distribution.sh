#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem-per-cpu=MaxMemPerCPU
#SBATCH --mem=MaxMemPerNode

/home/yppatel/anaconda3/bin/python3 find_score_distribution.py Clothing_Shoes_and_Jewelry_5.csv
/home/yppatel/anaconda3/bin/python3 find_score_distribution.py Electronics_5.csv
/home/yppatel/anaconda3/bin/python3 find_score_distribution.py Toys_and_Games_5.csv
/home/yppatel/anaconda3/bin/python3 find_score_distribution.py Video_Games_5.csv
