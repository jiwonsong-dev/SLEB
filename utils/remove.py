import torch
import numpy as np
from tqdm import tqdm

def remove_fn(model, kill_list):

    if kill_list is None:
      return model
      
    print(f"Removing layers... {kill_list}")
    if 'opt' in model.name.lower():
        return remove_opt(model, kill_list)
    elif 'llama' in model.name.lower():
        return remove_llama(model, kill_list)
    else:
        return None

def remove_opt(model, kill_list):

    kill_list.sort()
    
    while(len(kill_list)>0):
        
        del model.model.decoder.layers[kill_list[0]]
        del kill_list[0]
        for i in range(len(kill_list)):
            kill_list[i] -= 1
    
    return model

def remove_llama(model, kill_list):

    kill_list.sort()

    while(len(kill_list)>0):
        
        del model.model.layers[kill_list[0]]
        del kill_list[0]
        for i in range(len(kill_list)):
            kill_list[i] -= 1

    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn.layer_idx = i

    return model

def remove_single_opt(model, kill_list):
    
    killed = kill_list[0]
    del model.model.decoder.layers[kill_list[0]]
    del kill_list[0]
    for i in range(len(kill_list)):
        if  kill_list[i] > killed:
            kill_list[i] -= 1
    
    return model, kill_list

def remove_single_llama(model, kill_list):
    
    killed = kill_list[0]
    del model.model.layers[kill_list[0]]
    del kill_list[0]
    for i in range(len(kill_list)):
        if  kill_list[i] > killed:
            kill_list[i] -= 1

    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn.layer_idx = i
    
    return model, kill_list
