from utils.onoff_utils import onoff_opt, onoff_llama

def block_replace(model):
    if 'opt' in model.name.lower():
        model = onoff_opt.block_replace(model)
    elif 'llama' in model.name.lower():
        model = onoff_llama.block_replace(model)
    
    return model

def turn_off(model, block_idx):
    if 'opt' in model.name.lower():
        onoff_opt.turn_off(model, block_idx)
    elif 'llama' in model.name.lower():
        onoff_llama.turn_off(model, block_idx)

def turn_on(model, block_idx):
    if 'opt' in model.name.lower():
        onoff_opt.turn_on(model, block_idx)
    elif 'llama' in model.name.lower():
        onoff_llama.turn_on(model, block_idx)

def scan(model, num_blocks):

    if 'opt' in model.name.lower():
        onoff_opt.scan(model, num_blocks)
    elif 'llama' in model.name.lower():
        onoff_llama.scan(model, num_blocks)
    
