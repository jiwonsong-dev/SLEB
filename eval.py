import copy
from tqdm import tqdm
import fire
from typing import List

import torch

from utils.model_utils import get_llm
from utils.data_utils import *
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
from utils.remove import remove_fn

def eval(
        model_name: str = 'meta-llama/Llama-2-7b-hf',
        removal_list: List[int] = [14, 23, 11, 24, 10, 27, 15],
        save_model: bool = False,
        save_path: str = 'models/llama-2-7b_sleb_7_32',
        save_results: bool = True,
        result_path: str = "sleb_results/eval.txt",
        device: int = 0,
        eval_ppl: bool = True,
        eval_zeroshot: bool = False,
        eval_five_shot: bool = False,
        eval_mmlu: bool = False
    ):
    
    
    model = get_llm(model_name)
    print(f"Loaded Model: {model.name}")
    model.eval()
    
    original_removal_list = copy.deepcopy(removal_list)
    removal_list.sort()
    model = remove_fn(model, copy.deepcopy(removal_list))

    if save_model:
        model.save_pretrained(save_path)
        get_tokenizer(model_name).save_pretrained(save_path)
        print("Model and tokenizer successfully saved.")

    if eval_ppl:

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

        zeroshot_results = eval_zero_shot(model_name, copy.deepcopy(removal_list), zeroshot_tasks, parallelize=parallelize)
        zeroshot_results = zeroshot_results['results']

        for task in zeroshot_tasks:
            print(f"{task}: {zeroshot_results[task]}")
    
    if eval_five_shot:

        print(f"Starting Zero-shot tasks evaluation...")
        if '30b' or '66b' or '70b' in model_name:
            parallelize = True
        else:
            parallelize = False

        five_shot_tasks = ['piqa','winogrande','hellaswag','arc_challenge','arc_easy']

        five_shot_results = eval_zero_shot(model_name, copy.deepcopy(removal_list), five_shot_tasks, num_fewshot=5, parallelize=parallelize)
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

        mmlu_result = eval_zero_shot(model_name, copy.deepcopy(removal_list), mmlu_tasks, num_fewshot=5, parallelize=parallelize)
        mmlu_result = mmlu_result['results']

        for task in mmlu_tasks:
            print(f"{task}: {mmlu_result[task]}")
    
    if save_results:
        with open(result_path, 'a') as file:
            sentences = []
            sentences.append(f"Model Name: {model_name}\n")
            sentences.append(f"Block Removal Order: {original_removal_list}\n")
            
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
                sentences.append(f"MMLU results: {mmlu_tasks['mmlu']}\n")
                for task in mmlu_tasks:
                    sentences.append(f"{task}: {mmlu_result[task]}")
                sentences.append("\n")
            
            sentences.append("\n")
            
            for sentence in sentences:
                file.write(sentence)

if __name__ == "__main__":
    fire.Fire(eval)