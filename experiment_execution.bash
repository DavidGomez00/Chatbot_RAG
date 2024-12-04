#!/bin/bash

for chunk_str in "nat" "nat_sem"; do
  for model in "BGE-M3" "multilinguale5"; do
    for llm in "nemotron" "llama3.1:70b-instruct-fp16"; do
      python main.py --chunk_str "$chunk_str" --model "$model" --llm "$llm"
    done
  done
done
