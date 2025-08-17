#!/bin/sh

### MULTI-FACTOR (MAIN RUNS)

#python GenMol.py --protein DBH --target_size 5 --choice 3 --context True --model gpt-4o --final_k 100
#python GenMol.py --protein DBH --target_size 0 --choice 3 --context True --model gpt-4o --final_k 100

#python GenMol.py --protein JAK2 --target_size 5 --choice 3 --context True --model gpt-4o --final_k 100
#python GenMol.py --protein JAK2 --target_size 0 --choice 3 --context True --model gpt-4o --final_k 100

#python GenMol.py --protein DRD2 --target_size 5 --choice 3 --context True --model gpt-4o --final_k 100
#python GenMol.py --protein DRD2 --target_size 0 --choice 3 --context True --model gpt-4o --final_k 100


### SINGLE FACTOR (AFFINITY)
#python GenMol.py --protein JAK2 --target_size 5 --choice 1 --context True --model gpt-4o --final_k 100
#python GenMol.py --protein JAK2 --target_size 0 --choice 1 --context True --model gpt-4o --final_k 100

#python GenMol.py --protein DRD2 --target_size 5 --choice 1 --context True --model gpt-4o --final_k 100
#python GenMol.py --protein DRD2 --target_size 0 --choice 1 --context True --model gpt-4o --final_k 100

#python GenMol.py --protein DBH --target_size 5 --choice 1 --context True --model gpt-4o --final_k 100
#python GenMol.py --protein DBH --target_size 0 --choice 1 --context True --model gpt-4o --final_k 100


### FOR CLAUDE SONNET BASED MOLECULE GENERATION
#python GenMol_claude.py --protein JAK2 --target_size 5 --choice 3 --context True --model claude-3-5-sonnet-20241022 --final_k 100
#python GenMol_claude.py --protein JAK2 --target_size 0 --choice 3 --context True --model claude-3-5-sonnet-20241022 --final_k 100

#python GenMol_claude.py --protein DRD2 --target_size 5 --choice 3 --context True --model claude-3-5-sonnet-20241022 --final_k 100
#python GenMol_claude.py --protein DRD2 --target_size 0 --choice 3 --context True --model claude-3-5-sonnet-20241022 --final_k 100

#python GenMol_claude.py --protein DBH --target_size 5 --choice 3 --context True --model claude-3-5-sonnet-20241022 --final_k 100
#python GenMol_claude.py --protein DBH --target_size 0 --choice 3 --context True --model claude-3-5-sonnet-20241022 --final_k 100


### FOR ZINC GPT2 BASED MOLECULE GENERATION
#python GenMol_ZincGPT2.py --choice mf --protein JAK2 --final_k 100
#python GenMol_ZincGPT2.py --choice mf --protein DRD2 --final_k 100
#python GenMol_ZincGPT2.py --choice mf --protein DBH --final_k 100


### FOR NEW DATA: 4LRH
#python GenMol.py --protein 4LRH --target_size 5 --choice 3 --context True --model gpt-4o --final_k 100