import copy
from tqdm import tqdm
import fire
from typing import List

import torch
from peft import PeftModel

from utils.model_utils import get_llm
from utils.data_utils import *
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
from utils.remove import remove_fn

def eval(
        model_name: str = '/home/jiwonsong/LLM_pruning/sparsegptm/l2-7b',
        lora_weights: str = '/home/jiwonsong/SLEB/finetuning/adapters/home/jiwonsong/LLM_pruning/sparsegptm/l2-7b_e1_r8_len128',
        save_model: bool = False,
        save_path: str = 'models/llama-2-7b_sleb_7_32_ft',
        save_results: bool = True,
        result_path: str = "sleb_results/eval.txt",
        device: int = 0,
        eval_zeroshot: bool = True,
        seqlen: int = 2048
    ):
    
    
    model = get_llm(model_name)
    print(f"Loaded Model: {model.name}")
    
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
                                 lora_weights=lora_weights)
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