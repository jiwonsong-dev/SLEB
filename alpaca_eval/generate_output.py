import fire
from typing import List
import copy
import json
import os

import torch
import datasets

from utils.model_utils import get_llm
from utils.data_utils import get_tokenizer
from utils.remove import remove_fn

def get_after_slash(text):
    index = text.find('/')
    if index != -1:
        return text[index + 1:]
    else:
        return text
    
def generate(
            model_name: str = 'meta-llama/Llama-2-7b-hf',
            removal_list: List[int] = [14, 23, 11, 24, 10, 27, 15],
            save_path: str = 'alpaca_eval/outputs'
             ):
    
    run_name = f"{get_after_slash(model_name)}_{len(removal_list)}"
    save_path = os.path.join(save_path, run_name)
    
    model = get_llm(model_name)
    tokenizer = get_tokenizer(model_name)
    print(f"Loaded Model: {model.name}")
    model.eval()

    removal_list.sort()
    model = remove_fn(model, copy.deepcopy(removal_list))

    print(f"Starting output generation")

    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    for example in eval_set:
        
        input_len = len(example['instruction'])
        input_ids = tokenizer.encode(example['instruction'], return_tensors="pt").to(model.device)
        max_new_tokens=256

        
        output = model.generate(
                                input_ids,
                                min_new_tokens=1, 
                                max_new_tokens=max_new_tokens,
                                no_repeat_ngram_size=3,
                                )
            
        torch.cuda.empty_cache()
        example['output'] = tokenizer.decode(output[0], skip_special_tokens=True)[input_len:]

        with open(save_path, "a") as f:
            f.write(json.dumps(example, indent=4))
            f.write('\n')
            f.flush()

    print("JSON generation complete!\n")

if __name__ == '__main__':
    fire.Fire(generate)
