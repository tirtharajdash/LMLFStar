#!/bin/sh

#python GenMol.py --protein JAK2 --target_size 5 --choice 3 --context True --model gpt-4o --final_k 0

#python GenMol.py --protein DRD2 --target_size 0 --choice 3 --context True --model gpt-4o --final_k 50

#python GenMol.py --protein DBH --target_size 5 --choice 1 --context True --model gpt-4o --final_k 100
#python GenMol.py --protein DBH --target_size 5 --choice 2 --context True --model gpt-4o --final_k 100
python GenMol.py --protein DBH --target_size 5 --choice 3 --context True --model gpt-4o --final_k 100
