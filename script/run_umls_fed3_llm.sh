#!/bin/sh

python main.py --data_path ./data/umls-Fed3.pkl --name umls_fed3_llm \
              --setting LLM --mode test  --LLMModel together_ai --together_ai_model mistralai/Mixtral-8x7B-Instruct-v0.1