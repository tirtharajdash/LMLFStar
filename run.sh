#!/bin/sh

#python GenMol.py --protein DBH --target_size 5 --choice 3 --context True --model gpt-4o --final_k 100
#python GenMol.py --protein DBH --target_size 0 --choice 3 --context True --model gpt-4o --final_k 100

#python GenMol.py --protein JAK2 --target_size 5 --choice 3 --context True --model gpt-4o --final_k 100
#python GenMol.py --protein JAK2 --target_size 0 --choice 3 --context True --model gpt-4o --final_k 100

#python GenMol.py --protein DRD2 --target_size 5 --choice 3 --context True --model gpt-4o --final_k 100
#python GenMol.py --protein DRD2 --target_size 0 --choice 3 --context True --model gpt-4o --final_k 100

### FOR NEW DATA: 4LRH
python GenMol.py --protein 4LRH --target_size 5 --choice 3 --context True --model gpt-4o --final_k 100
