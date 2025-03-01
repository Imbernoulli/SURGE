from rl import rl_eval
from fl import fl_eval, jaccard_similarity
from sc import evaluate_output, extract_output
import json
from tqdm import tqdm

def re_errors(data):
    for d in data:
        if "custom_categoey" in d:
            d["custom_category"] = d["custom_categoey"]
        if d["custom_category"] == 1 and "difficulty" in d["custom_features"]:
            d["custom_category"] = 2
        if d["custom_category"] == 3:
            if(d["custom_features"]["repo"] == "PlotNeuralNet"):
                d["custom_features"]["language"] = "Python"
                d["custom_features"]["task"] = "plotneurnalnet"
                continue
            elif(d["custom_features"]["run_instruction"] == "python 24_game.py"):
                d["custom_features"]["task"] = "24_game"
                continue
            elif(d["custom_features"]["repo"] == "https://github.com/psychopurp/sudoku-player"):
                continue
            else:
                if d["custom_features"]["run_instruction"] == "make":
                    d["custom_features"]["task"] = "TUOJ,cow"
                else:
                    d["custom_features"]["task"] = "TUOJ,car"
                d["custom_features"]["language"] = "CPP"
    return data

suffixes = ['sys_0shot', 'sys_3shot', 'sys_cot']

with open("./../allll.json", "r") as f:
    data = re_errors(json.load(f))

with open("./../data/allll3.json", "r") as f:
    temp_data = re_errors(json.load(f))

models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct", 
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct", 
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        
        "models/Meta-Llama-3.1-8B-Instruct", 
        "meta-llama/llama-3.3-70b-instruct_"]

for d in data:
    for td in temp_data:
        if d["question"] == td["question"]:
            for key1 in models:
                for key2 in suffixes:
                    d[key1 + key2] = td[key1 + key2]

with open("./../data/allll.json", "r") as f:
    temp_data = re_errors(json.load(f))

models =    ["claude-3-5-sonnet-20241022_",   
    "gpt-4o-2024-08-06_",
    "gpt-4o-mini-2024-07-18_",
    "qwen-max-latest_",
    "deepseek-v3_"]

for d in data:
    for td in temp_data:
        if d["question"] == td["question"]:
            for key1 in models:
                for key2 in suffixes:
                    d[key1 + key2] = td[key1 + key2]

def evall(data, key):
    if data["custom_category"] in [1, 2]:
        return data["gt"] == extract_output(data[key])
    elif data["custom_category"] in [4, 5]:
        return evaluate_output(data[key], data["gt"], data["custom_features"]["fid"])
    elif data["custom_category"] in [6, 7]:
        return jaccard_similarity(data["gt"], extract_output(data[key]))
    elif data["custom_category"] == 3:
        return rl_eval(data, key)
    else:
        return fl_eval(data[key], data["gt"])

model_mapping = {
    # Llama 3.1 models
    "models/Meta-Llama-3.1-8B-Instruct": "LLaMA-3.1-8B-Instruct",
    "models/Meta-Llama-3.1-70B-Instruct": "LLaMA-3.1-70B-Instruct",
    "meta-llama/llama-3.3-70b-instruct_": "LLaMA-3.3-70B-Instruct",

    # Qwen 2.5 base models
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen-2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen-2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen-2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen-2.5-7B-Instruct",
    "models/Qwen2.5-14B-Instruct": "Qwen-2.5-14B-Instruct",
    "models/Qwen2.5-32B-Instruct": "Qwen-2.5-32B-Instruct",
    "models/Qwen2.5-72B-Instruct": "Qwen-2.5-72B-Instruct",

    # Qwen 2.5 Coder models
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": "Qwen-2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "Qwen-2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct": "Qwen-2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen-2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct": "Qwen-2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen-2.5-Coder-32B-Instruct",

  "claude-3-5-sonnet-20241022_": "Claude-3.5-Sonnet",
  "gpt-4o-2024-08-06_": "GPT-4o",
  "gpt-4o-mini-2024-07-18_": "GPT-4o-Mini",
  "qwen-max-latest_": "Qwen-Max",
  "deepseek-v3_": "DeepSeek-V3"
}

models = [
    "claude-3-5-sonnet-20241022_", 
    "deepseek-v3_",  
    "gpt-4o-2024-08-06_",
    "gpt-4o-mini-2024-07-18_",
    "qwen-max-latest_",
    

        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct", 
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "models/Qwen2.5-14B-Instruct",
        "models/Qwen2.5-32B-Instruct",
        "models/Qwen2.5-72B-Instruct",
        
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct", 
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        
        "models/Meta-Llama-3.1-8B-Instruct",
        
        
        "models/Meta-Llama-3.1-70B-Instruct",

        "meta-llama/llama-3.3-70b-instruct_"
        

    ]

if __name__ == "__main__":
    for d in tqdm(data):    
        for key1 in models:
            for key2 in suffixes:
                    d[key1 + key2 + "eval"] = evall(d, key1 + key2)
                    if not d[key1 + key2 + "eval"]:
                        d[key1 + key2 + "eval"] = 0

    with open("allll_eval.json", "w") as f:
        f.write(json.dumps(data, indent = 4))