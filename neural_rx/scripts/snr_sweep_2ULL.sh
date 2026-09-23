#!/bin/bash

for model in nrx_large_3dmrs_13sym1_2.onnx datalake_j61_2ULL_rt.onnx nrx_large_3dmrs_13sym1.onnx datalake_j61_2ULL.onnx; do
    for X in {0..25}; do
        python eval_neural_receiver_from_datalake.py \
            --limit 1000 \
            --model "$model" \
            --timestamps j61_2ULL.pkl \
            --add-noise-snr "$X" \
            --db 8 \
	    --use-onnx
    done
done

python eval_neural_receiver_from_datalake.py --db 8 --model nrx_large_3dmrs_13sym1_2.onnx --timestamps j61_2ULL.pkl --limit 1000 --use-onnx
