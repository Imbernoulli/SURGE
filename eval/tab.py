import json
from collections import defaultdict
from main import models, suffixes, model_mapping

import json
from collections import defaultdict

# Read the json file
with open("allll_eval.json", "r") as f:
    data = json.load(f)

# Function to determine language category for ML
def get_ml_lang_category(features):
    if "language" not in features:
        return "Others"
    lang = features["language"].lower()
    if lang == "cpp":
        return "CPP"
    elif lang == "rust":
        return "Rust" 
    elif lang == "python":
        return "Python"
    elif lang == "julia":
        return "Julia"
    elif lang == "java":
        return "Java"
    return "Others"

# Function to determine language category for BG
def get_bg_lang_category(features):
    if "language" not in features:
        return None
    lang = features["language"].lower()
    if lang == "cpp":
        return "CPP"
    elif lang == "java":
        return "Java"
    elif lang == "python3":
        return "Python"
    return None

# Initialize results dictionary
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Calculate results for each model and category
for d in data:
    cat = d["custom_category"]
    
    for model in models:
        for suffix in ["sys_0shot", "sys_cot", "sys_3shot"]:
            key = model + suffix + "eval"
            if key in d:
                if cat == 1:
                    lang = get_ml_lang_category(d["custom_features"])
                    results[suffix][model][f"ML_{lang}"].append(d[key])
                elif cat == 6:
                    lang = get_bg_lang_category(d["custom_features"])
                    if lang:  # Only add if language is one of CPP, Java, Python
                        results[suffix][model][f"BG_{lang}"].append(d[key])
                else:
                    results[suffix][model][cat].append(d[key])

# Function to calculate average
def calc_avg(values):
    return (sum(values) / len(values) * 100) if values else 0

# Function to generate table section
def generate_section(suffix_data, suffix_name):
    lines = []
    lines.append("\\multicolumn{15}{c}{\\textit{" + suffix_name + "}}                                                                        \\\\ \\midrule")
    
    for model in models:
        model_name = model_mapping[model]
        
        # Calculate ML scores by language
        ml_scores = []
        for lang in ["CPP", "Rust", "Java", "Python", "Julia", "Others"]:
            score = calc_avg(suffix_data[model][f"ML_{lang}"])
            ml_scores.append(f"${score:.2f}$")
            
        # Calculate BG scores by language
        bg_scores = []
        for lang in ["CPP", "Java", "Python"]:
            score = calc_avg(suffix_data[model][f"BG_{lang}"])
            bg_scores.append(f"${score:.2f}$")
            
        # Calculate other category scores
        other_scores = []
        for cat in [2, 3, 4, 5, 7, 8]:  # Skip 6 (BG) as it's handled separately
            score = calc_avg(suffix_data[model][cat])
            other_scores.append(f"${score:.2f}$")
            
        # Calculate total average
        all_scores = [calc_avg(suffix_data[model][f"ML_{lang}"]) for lang in ["CPP", "Rust", "Python", "Julia", "Java", "Others"]]
        all_scores.extend([calc_avg(suffix_data[model][f"BG_{lang}"]) for lang in ["CPP", "Java", "Python"]])
        all_scores.extend([calc_avg(suffix_data[model][cat]) for cat in [2, 3, 4, 5, 7, 8]])
        valid_scores = [score for score in all_scores if score != 0]  # Only consider non-zero scores
        total = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
        line = f"\\texttt{{" + model_name + "} & "
        line += " & ".join(ml_scores)
        line += " & " + " & ".join(other_scores[:4])  # CL, RL, SC, TC
        line += " & " + " & ".join(bg_scores)  # BG by language
        line += " & " + " & ".join(other_scores[4:])  # DR, FL
        line += f" & ${total:.2f}$ \\\\"
        lines.append(line)
    
    return lines

# Generate the complete table
table_lines = [
    "\\begin{table*}[!H]",
    "\\centering",
    "\\caption{Results of different models and different prompting strategies.}",
    "\\resizebox{0.9\\textwidth}{!}{\\begin{tabular}{l*{15}{S}}",
    "\\toprule",
    "\\textbf{Model} & \\multicolumn{6}{c}{\\textbf{ML}} & \\textbf{CL} & \\textbf{RL} & \\textbf{SC} & \\textbf{TC} & \\multicolumn{3}{c}{\\textbf{BG}} & \\textbf{DR} & \\textbf{FL} & \\textbf{Avg.} \\\\ \\cmidrule(lr){2-7} \\cmidrule(lr){11-13}",
    "& \\textbf{CPP} & \\textbf{Rust} & \\textbf{Python} & \\textbf{Julia} & \\textbf{Java} & \\textbf{Others} &&&&& \\textbf{CPP} & \\textbf{Java} & \\textbf{Python} &&& \\\\ \\midrule"
]

# Add sections
table_lines.extend(generate_section(results["sys_0shot"], "0-shot"))
table_lines.append("\\midrule")
table_lines.extend(generate_section(results["sys_cot"], "0-shot Chain-of-Thought"))
table_lines.append("\\midrule")
table_lines.extend(generate_section(results["sys_3shot"], "few-shot Chain-of-Thought"))

table_lines.extend([
    "\\bottomrule",
    "\\end{tabular}}",
    "\\end{table*}"
])

# Write to file
with open("table.tex", "w") as f:
    f.write("\n".join(table_lines))