#!/bin/bash

values=(100 500 1000 5000 10000 20000 50000 100000)

for X in "${values[@]}"; do
    python eval_neural_receiver_from_datalake.py \
        --limit 2000 \
        --model "datalake_13sym_1tx_${X}_iter.plan" \
        --ue 1 \
        --timestamps timestamps1.pkl
done

