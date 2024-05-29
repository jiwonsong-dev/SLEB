import fire
import copy
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn

from utils.model_utils import get_llm
from utils.onoff_utils.onoff import block_replace, turn_off, turn_on
from utils.data_utils import *
from utils.block_remove import block_remove
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot

@torch.no_grad()
def get_loss(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
  
    # List to store negative log likelihoods
    losses = []
    #print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        loss = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        losses.append(loss)

    # Compute sum of negative log_likelihood
    loss_sum = torch.stack(losses).sum()

    return loss_sum.item()


def sleb(
        model_name: str = 'meta-llama/Llama-2-7b-hf',
        num_blocks: int = 32,
        num_remove_blocks: int = 7,
        early_barrier: int = 1,
        latter_barrier: int = 1,
        seed: int = 0,
        nsamples: int = 128,
        result_folder: str = 'sleb_results',
        result_file: str = 'sleb_results.txt',
        dataset: str = 'wikitext2',
        eval_ppl: bool = True,
        eval_zeroshot: bool = False
):
    alive_list = [i for i in range(num_blocks)]
    removal_list = []

    model = get_llm(model_name)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print(f"Loaded Model: {model.name}")
    
    # replace
    model = block_replace(model)
    model.eval()

    dataloader = get_trainloaders(dataset,
                                  nsamples=nsamples,
                                 seed=seed,
                                 model=model_name,
                                 )
    print(f"Dataloader({dataset}) loaded.")

    # check start time
    start_point = time.time()
    for i in range(num_remove_blocks):

        phase_start_point = time.time()
        print(f"Phase {i+1} of {num_remove_blocks}")

        min_loss = 1e99
        min_loss_idx = -1

        search_bound = num_blocks - i

        for j in range(early_barrier, search_bound-latter_barrier):

            # kill j-th alive block
            turn_off(model, alive_list[j])

            loss = get_loss(model, dataloader, bs=1, device=torch.device("cuda:0"))
            torch.cuda.empty_cache()
            
            if loss < min_loss:
                min_loss = loss
                min_loss_idx = j

            print(
                f"[Block {j} (Original block {alive_list[j]}) removed] Loss={loss:.3f}, Current Min Loss={min_loss:.3f} / Layer {alive_list[min_loss_idx]}"
            )
            # unkill j-th alive block
            turn_on(model, alive_list[j])

        
        phase_finish_point = time.time()
        phase_time_elapsed = phase_finish_point -  phase_start_point

        # remove block causing the least snlls increase
        print(f"Phase_time_elapsed (s): {phase_time_elapsed}")
        print(f"[SELECTED block {min_loss_idx} (Originally block {alive_list[min_loss_idx]})] Loss={min_loss:.3f}")      

        turn_off(model, alive_list[min_loss_idx])
        removal_list.append(alive_list[min_loss_idx])
        print(f"Current Block Removal List: {removal_list}")
        del alive_list[min_loss_idx]
    
    finish_point = time.time()
    time_elapsed = finish_point - start_point

    print(
        f"Time_Elapsed: {time_elapsed}\n"
        f"Model Name: {model_name}\n"
        f"# Total Blocks: {num_blocks}\n"
        f"# Remove Blocks: {num_remove_blocks}\n"
        f"Dataset: {dataset}\n"
        f"Seed: {seed}\n" 
        f"Barriers: early {early_barrier} / latter {latter_barrier}\n" 
        f"Block Removal Order: {removal_list}\n"
    )

    if eval_ppl:
        print(f"Starting PPL evaluation...")
        model = block_remove(model, copy.deepcopy(removal_list))
        model.config.use_cache = use_cache

        w2_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2')
        print(f"WikiText-2 PPL = {w2_ppl:.2f}")

        c4_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='c4')
        print(f"C4 PPL = {c4_ppl:.2f}")

    if eval_zeroshot:
        print(f"Starting Zero-shot tasks evaluation...")
        if '30b' or '66b' or '70b' in model_name:
            parallelize = True
        else:
            parallelize = False

        tasks = ['piqa','winogrande','hellaswag','arc_challenge','arc_easy']

        results = eval_zero_shot(model_name, copy.deepcopy(removal_list), tasks, parallelize=parallelize)
        results = results['results']

        for task in tasks:
            print(f"{task}: {results[task]}")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    result_path = os.path.join(result_folder, result_file)

    with open(result_path, 'a') as file:
        sentences = []
        sentences.append(f"Time Elapsed: {time_elapsed}\n")
        sentences.append(f"Model Name: {model_name}\n")
        sentences.append(f"# Total Blocks: {num_blocks}\n")
        sentences.append(f"# Remove Blocks: {num_remove_blocks}\n")
        sentences.append(f"Dataset: {dataset}\n")
        sentences.append(f"Seed: {seed}\n") 
        sentences.append(f"Barriers: early {early_barrier} / latter {latter_barrier}\n")
        sentences.append(f"Block Removal Order: {removal_list}\n")

        if eval_ppl:
            sentences.append(f"WikiText-2 PPL = {w2_ppl:.2f}\n")
            sentences.append(f"C4 PPL = {c4_ppl:.2f}\n")

        if eval_zeroshot:
            sentences.append(f"Zero-shot results: \n")
            for task in tasks:
                sentences.append(f"{task}: {results[task]}\n")
        sentences.append("\n")

                             
        for sentence in sentences:
            file.write(sentence)

if __name__ == "__main__":
    fire.Fire(sleb)