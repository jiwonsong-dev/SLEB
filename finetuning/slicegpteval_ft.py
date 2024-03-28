import copy
import os
from tqdm import tqdm
import fire
from typing import List

import torch
from peft import PeftModel

from utils.model_utils import get_llm
from utils.data_utils import *
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils

def eval(
        model_name: str = 'meta-llama/Llama-2-7b-hf',
        sparsity: float = 0.25,
        lora_weights: str = '',
        save_model: bool = False,
        save_path: str = 'models/llama-2-7b_sleb_7_32_ft',
        save_results: bool = True,
        result_path: str = "sleb_results/eval.txt",
        device: int = 0,
        eval_zeroshot: bool = True,
        seqlen: int = 2048,
    ):
    
    lora_weights = lora_weights + str(sparsity)

    if '7b' in model_name:
        designator = '7b'
    elif '13b' in model_name:
        designator = '13b'
    else:
        designator = '70b'
    
    if sparsity == 0.2:
        sliced_model_path = f'/home/jiwonsong/LLM_pruning/TransformerCompression/experiments/dir/Llama-2-{designator}-hf_0.2.pt'
    elif sparsity == 0.25:
        sliced_model_path = f'/home/jiwonsong/LLM_pruning/TransformerCompression/experiments/dir/Llama-2-{designator}-hf_0.25.pt'
    else:
        sliced_model_path = f'/home/jiwonsong/LLM_pruning/TransformerCompression/experiments/dir/Llama-2-{designator}-hf_0.3.pt'

    print(f"Loading sliced {model_name} model from {sliced_model_path} with sparsity {sparsity}")
    model_adapter, _ = hf_utils.load_sliced_model(
            model_name, sliced_model_path, sparsity, os.getenv('HF_TOKEN', None)
        )
    model = model_adapter.model
    model = model.cuda()
    model.name = model_name
    print(f"Loaded Sliced Model: {model.name} with sparsity {sparsity}")
    
    model.eval()

    model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    print(f"Loaded LoRA weights.")

    model = model.merge_and_unload(progressbar=True)
    print(f"Merged LoRA adapters to the model.")

    if save_model:
        model.save_pretrained(save_path)
        get_tokenizer(model_name).save_pretrained(save_path)
        print("Model and tokenizer successfully saved.")

    model.seqlen = seqlen
    
    print(f"Starting PPL evaluation...")
    ppl_list = {}
    test_datasets = ['wikitext2', 'c4']
    for dataset in test_datasets:
        ppl = load_and_eval_ppl(model, device, dataset=dataset)

        print(f"{dataset} perplexity = {ppl:.2f}")

        ppl_list[dataset] = ppl
    

    del model
    torch.cuda.empty_cache()

    if eval_zeroshot:
        print(f"Starting Zero-shot tasks evaluation...")
        if '30b' or '66b' or '70b' in model_name:
            parallelize = True
        else:
            parallelize = False

        tasks = ['piqa','winogrande','hellaswag','arc_challenge','arc_easy']

        results = eval_zero_shot(model_name, 
                                 [], 
                                 tasks, 
                                 parallelize=parallelize,
                                 lora_weights=lora_weights,
                                 slice=True,
                                 sliced_model_sparsity=sparsity,
                                 sliced_model_path=sliced_model_path,
                                 )
        results = results['results']

        for task in tasks:
            print(f"{task}: {results[task]}")
    
    if save_results:
        with open(result_path, 'a') as file:
            sentences = []
            sentences.append(f"Model Name: {model_name}\n")
            
            if eval_zeroshot:
                sentences.append(f"WikiText-2 PPL: {ppl_list['wikitext2']:.2f}\n")
                sentences.append(f"C4 PPL: {ppl_list['c4']:.2f}\n")
                sentences.append(f"Zero-shot results: \n")
                for task in tasks:
                    sentences.append(f"{task}: {results[task]}\n")
                sentences.append("\n")
            else:
                sentences.append(f"WikiText-2 PPL: {ppl_list['wikitext2']:.2f} ")
                sentences.append(f"C4 PPL: {ppl_list['c4']:.2f}\n\n")
            
            for sentence in sentences:
                file.write(sentence)

if __name__ == "__main__":
    fire.Fire(eval)