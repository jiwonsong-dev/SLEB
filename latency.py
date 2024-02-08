import torch
import fire
import numpy as np
from utils import remove, model_utils, latency_utils

def test_latency(
    model_name='meta-llama/Llama-2-7b-hf',
    skip_ratio=0.2,
    generation=False,
    result_path='sleb_results/latency.txt'
):
    gpu_num = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    model = model_utils.get_llm(model_name)
    model.eval()
    num_of_layers = model.config.num_hidden_layers
    skip_num = int(np.ceil(num_of_layers * skip_ratio))
    remove_list = [i+1 for i in range(skip_num)]
    
    print("==================================================")
    print("Experiment Environment")
    print(f"Current GPU: {gpu_name}")
    print(f"# GPU: {str(gpu_num)}")
    print(f"Model Name: {model_name}")
    print(f"Infernce type : {'Token Generation' if generation else 'Prompt Processing'}")
    print("==================================================")

    # latency for dense model
    dense_latency = latency_utils.test_latency(model, generation)
    print(f"Dense Latency: {dense_latency:.2f}ms")

    # latency for sleb model
    if 'llama' in model_name:
        model = remove.remove_llama(model, remove_list)
    elif 'opt' in model_name:
        model = remove.remove_opt(model, remove_list)
    sleb_latency = latency_utils.test_latency(model, generation)
    print(f"SLEB {skip_ratio} Latency: {sleb_latency:.2f}ms")

    # speedup
    speedup = dense_latency / sleb_latency
    print(f"Speedup: x{speedup:.2f}")
    print("==================================================")

    # save log
    with open(result_path, "a") as file: 
        file.write(f"Current GPU: {gpu_name}")
        file.write(", ")
        file.write(f"# GPU: {str(gpu_num)}")
        file.write(", ")
        file.write(f"Model Name: {model}")
        file.write(", ")
        file.write(f"Infernce type : {'Token Generation' if generation else 'Prompt Processing'}")
        file.write(", ")
        file.write(f"Dense Latency: {dense_latency:.2f}ms")
        file.write(", ")
        file.write(f"SLEB {skip_ratio} Latency: {sleb_latency:.2f}ms")
        file.write(", ")
        file.write(f"Speedup: x{speedup:.2f}")
        file.write("\n")

if __name__ == '__main__':
    fire.Fire(test_latency)