import fire
import copy
import time
from tqdm import tqdm

import torch
import torch.nn as nn

from utils.model_utils import get_llm
from utils.onoff_utils.onoff import replace, turn_off, turn_on
from utils.data_utils import *
from utils.remove import remove
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot

@torch.no_grad()
def nlls(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
  
    # List to store negative log likelihoods
    nlls = []
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
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute sum of negative log_likelihood
    #ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    sum = torch.stack(nlls).sum()

    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return sum.item()


def sleb(
        model_name: str = 'facebook/opt-6.7b',
        num_blocks: int = 32,
        num_remove_blocks: int = 7,
        early_barrier: int = 1,
        latter_barrier: int = 1,
        seed: int = 0,
        nsamples: int = 128,
        result_path: str = 'sleb_results/sleb.txt',
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
    model = replace(model)
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
        print(f"phase {i+1} of {num_remove_blocks}")

        min_snlls = 1e99
        min_snlls_idx = -1

        search_bound = num_blocks - i

        for j in range(early_barrier, search_bound-latter_barrier):

            # kill j-th alive block
            turn_off(model, alive_list[j])

            snlls = nlls(model, dataloader, bs=1, device=torch.device("cuda:0"))
            torch.cuda.empty_cache()
            
            if snlls < min_snlls:
                min_snlls = snlls
                min_snlls_idx = j

            print(
                f"block {j} removed (original block {alive_list[j]}) => snlls={snlls:.3f}\n"
                f"current min snlls_idx orignal block {alive_list[min_snlls_idx]} => min_snlls={min_snlls:.3f}"
            )
            # unkill j-th alive block
            turn_on(model, alive_list[j])

        
        phase_finish_point = time.time()
        phase_time_elapsed = phase_finish_point -  phase_start_point

        # remove block causing the least snlls increase
        print(f"phase_time_elapsed {phase_time_elapsed}")
        print(f"SELECTED block {min_snlls_idx} (original lyaer {alive_list[min_snlls_idx]} to be removed) => ")        
        print(f"snlls={min_snlls}")

        turn_off(model, alive_list[min_snlls_idx])
        removal_list.append(alive_list[min_snlls_idx])
        print(f"current skip list: {removal_list}")
        del alive_list[min_snlls_idx]
    
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
        model = remove(model, copy.deepcopy(removal_list))
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