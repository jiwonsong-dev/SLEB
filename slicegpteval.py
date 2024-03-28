import copy
from tqdm import tqdm
import fire
from typing import List
import os

import torch

from utils.model_utils import get_llm
from utils.data_utils import *
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
from utils.remove import remove_fn

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils


def eval(
        model_name: str = 'meta-llama/Llama-2-7b-hf',
        sparsity: float = 0.25,
        save_model: bool = False,
        save_path: str = 'models/llama-2-7b_sleb_7_32',
        save_results: bool = True,
        result_path: str = "sleb_results/eval.txt",
        device: int = 0,
        eval_ppl: bool = True,
        eval_zeroshot: bool = False,
        eval_five_shot: bool = False,
        eval_mmlu: bool = False,
        seqlen: int = 2048,
    ):

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

    

    if eval_ppl:
        
        print(f"Loading sliced {model_name} model from {sliced_model_path} with sparsity {sparsity}")
        model_adapter, _ = hf_utils.load_sliced_model(
                model_name, sliced_model_path, sparsity, os.getenv('HF_TOKEN', None)
            )
        model = model_adapter.model

        if '30b' or '66b' or '70b' in model_name:
            gpu_utils.distribute_model(model_adapter)
        else:
            model = model.cuda()
            
        model.name = model_name
        print(f"Loaded Sliced Model: {model.name} with sparsity {sparsity}")
        

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

        zeroshot_tasks = ['piqa','winogrande','hellaswag','arc_challenge','arc_easy']

        zeroshot_results = eval_zero_shot(model_name, 
                                          [],
                                        zeroshot_tasks, 
                                        parallelize=parallelize,
                                        slice=True,
                                        sliced_model_sparsity=sparsity,
                                        sliced_model_path=sliced_model_path,
                                        )
        zeroshot_results = zeroshot_results['results']

        for task in zeroshot_tasks:
            print(f"{task}: {zeroshot_results[task]}")
    
    if eval_five_shot:

        print(f"Starting Five-shot tasks evaluation...")
        if '30b' or '66b' or '70b' in model_name:
            parallelize = True
        else:
            parallelize = False

        five_shot_tasks = ['piqa','winogrande','hellaswag','arc_challenge','arc_easy']

        five_shot_results = eval_zero_shot(model_name, 
                                           [], 
                                           five_shot_tasks, 
                                           num_fewshot=5, 
                                           parallelize=parallelize,
                                           slice=True,
                                           sliced_model_sparsity=sparsity,
                                           sliced_model_path=sliced_model_path,
                                           )
        five_shot_results = five_shot_results['results']

        for task in five_shot_tasks:
            print(f"{task}: {five_shot_results[task]}")

    if eval_mmlu:

        print(f"Starting MMLU (5-shot) evaluation...")
        if '30b' or '66b' or '70b' in model_name:
            parallelize = True
        else:
            parallelize = False

        mmlu_tasks = ['mmlu']

        mmlu_result = eval_zero_shot(model_name, [], mmlu_tasks, num_fewshot=5, parallelize=parallelize)
        mmlu_result = mmlu_result['results']

        for task in mmlu_tasks:
            print(f"{task}: {mmlu_result[task]}")
    
    if save_results:
        with open(result_path, 'a') as file:
            sentences = []
            sentences.append(f"Model Name: {model_name}\n")
            sentences.append(f"Sparsity: {sparsity}\n")
            
            if eval_ppl:
                sentences.append(f"WikiText-2 PPL: {ppl_list['wikitext2']:.2f}\n")
                sentences.append(f"C4 PPL: {ppl_list['c4']:.2f}\n")

            if eval_zeroshot:
                sentences.append(f"Zero-shot results: \n")
                for task in zeroshot_tasks:
                    sentences.append(f"{task}: {zeroshot_results[task]}\n")
                sentences.append("\n")

            if eval_five_shot:
                sentences.append(f"Five-shot results: \n")
                for task in five_shot_tasks:
                    sentences.append(f"{task}: {five_shot_results[task]}\n")
                sentences.append("\n")

            if eval_mmlu:
                sentences.append(f"MMLU results: \n")
                for task in mmlu_tasks:
                    sentences.append(f"{task}: {mmlu_result[task]}")
                sentences.append("\n")
            
            sentences.append("\n")
            
            for sentence in sentences:
                file.write(sentence)

if __name__ == "__main__":
    fire.Fire(eval)