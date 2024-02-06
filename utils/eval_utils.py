# Import necessary modules
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn

# Import get_loaders function from data module within the same director
from utils.data_utils import *
import fnmatch


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
@torch.no_grad()
def load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2', testloader=None, tokenizer=None):
    # Print status
    print(f"Evaluating on {dataset}")

    # Get the test loader
    if testloader is None:
        if tokenizer is None:
            tokenizer = get_tokenizer(model.name)

        _, testloader = get_loaders(
            dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
        )
        print(f"Dataset Loaded.")

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl(model, testloader, 1, device)
    return ppl_test 

@torch.no_grad()
def eval_ppl(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0,nsamples,bs)):

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

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()

torch.no_grad()
def eval_zero_shot(model_name, removal_list, task_list=['piqa','winogrande','hellaswag','arc_challenge','arc_easy'], 
        num_fewshot=0, parallelize=False):
    
    from lm_eval import tasks, evaluator, utils
    task_manager = tasks.TaskManager(include_path='lm-evaluation-harness/lm_eval/tasks')
 
    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
    task_missing = [
        task
        for task in task_list
        if task not in task_names and "*" not in task
        ]  # we don't want errors if a wildcard ("*") task name was used
    
    
    model_args = f"pretrained={model_name},"
    if parallelize:
        model_args = f"pretrained={model_name},parallelize=True"

    if len(removal_list)>0:
        remove = True
    else:
        remove = False

    results = evaluator.simple_evaluate(
        model='hf',
        model_args=model_args,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size='auto',
        max_batch_size=None,
        device='cuda:0',
        use_cache=None,
        limit=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=False,
        gen_kwargs=None,
        task_manager=task_manager,
        remove = remove,
        removal_list = removal_list,
    )

    return results 