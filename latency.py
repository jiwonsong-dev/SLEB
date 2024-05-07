import torch
import fire
import numpy as np
import os

from SLEB.utils.block_remove import block_remove
from utils import model_utils, latency_utils

def run(
    model_name: str = 'meta-llama/Llama-2-7b-hf',
    removal_ratio: float = 0.2,
    generation: bool = False,
    result_folder: str ='sleb_results',
    result_file: str = 'latency.txt'
):
    gpu_num = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    model = model_utils.get_llm(model_name)
    model.eval()
    num_of_blocks = model.config.num_hidden_layers
    num_removal = int(np.ceil(num_of_blocks * removal_ratio))
    removal_list = [i+1 for i in range(num_removal)]
    
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
    model = block_remove(model, removal_list)
    sleb_latency = latency_utils.test_latency(model, generation)
    print(f"SLEB {removal_ratio} Latency: {sleb_latency:.2f}ms")

    # speedup
    speedup = dense_latency / sleb_latency
    print(f"Speedup: x{speedup:.2f}")
    print("==================================================")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_path = os.path.join(result_folder, result_file)

    # save log
    with open(result_path, "a") as file: 
        file.write(f"Current GPU: {gpu_name}")
        file.write(", ")
        file.write(f"# GPU: {str(gpu_num)}")
        file.write(", ")
        file.write(f"Model Name: {model_name}")
        file.write(", ")
        file.write(f"Infernce type : {'Token Generation' if generation else 'Prompt Processing'}")
        file.write(", ")
        file.write(f"Dense Latency: {dense_latency:.2f}ms")
        file.write(", ")
        file.write(f"SLEB {removal_ratio} Latency: {sleb_latency:.2f}ms")
        file.write(", ")
        file.write(f"Speedup: x{speedup:.2f}")
        file.write("\n")

if __name__ == '__main__':
    fire.Fire(run)