#!/bin/bash

for model in nrx_large_3dmrs_13sym1.plan nrx_large_3dmrs_13sym1_2.plan nrx_datalake_j61.plan nrx_datalake_j61_rt.plan; do
    for X in {0..25}; do
        python eval_neural_receiver_from_datalake.py \
            --limit 2000 \
            --model "$model" \
            --cell-id 51 \
            --ue 1 \
            --timestamps timestamps1.pkl \
            --add-noise-snr "$X" \
            --db 0
    done
done

