# SLEB
Official Implementation of SLEB: Streamlining LLMs through Redundancy Verification and  Elimination of Transformer Blocks

## Installation

    git clone https://github.com/leapingjagg-dev/SLEB.git
    conda create -n sleb python==3.10
    cd sleb
    pip install -r requirements.txt
    cd lm-evaluation-harness
    pip install -e .
    cd ..
    mkdir sleb_results

## Examples

To find 20% of blocks to remove and evaluate perplexity and zero-shot tasks performances (LLaMA-2-70B):

    python -m sleb --model_name meta-llama/Llama-2-70b-hf --num_blocks 80 --num_remove_blocks 16 --eval_ppl True --eval_zeroshot True

To evaluate performances of a model with designated blocks removed:

    python -m eval --model_name facebook/opt-13b --removal_list '[5, 4, 9, 2, 14, 25, 34, 10]' --eval_zeroshot True

To evaluate speedup of a 20% removed model compared to a dense model:

    python3 -m latency --model_name meta-llama/Llama-2-7b-hf --generation 