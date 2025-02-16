import json
import jsonlines
from scipy.stats import spearmanr
import numpy as np
import re
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Error:
    line: int
    message: str
def save_overall_results(model, output_key, category, total, passed):
    with open('overall_results.jsonl', 'a') as f:
        accuracy = passed/total*100.0
        f.write(json.dumps({"model": model, "output_key": output_key, "category": category, "total": total, "passed": passed, "accuracy": f"{accuracy:.2f}%"}) + '\n')







def parse_json_result(json_str: str) -> Tuple[bool, List[Error]]:
    """Parse JSON result string into pass status and list of errors"""
    try:
        result = json.loads(json_str)
        errors = []
        for error in result.get("errors", []):
            pos = error["pos"]
            errors.append(Error(
                line=pos["line"],
                message=error["data"]
            ))
        return result.get("pass", False), errors
    except:
        return False, []

def jaccard_similarity2(str1: str, str2: str) -> float:
    """Calculate Jaccard similarity between two strings"""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def calculate_fl_score(pred_json: str, truth_json: str) -> float:
    """Calculate the FL score between predicted and ground truth results"""
    pred_pass, pred_errors = parse_json_result(pred_json)
    truth_pass, truth_errors = parse_json_result(truth_json)
    
    # If both pass or both fail with no errors
    if pred_pass and truth_pass:
        return 1.0
    elif pred_pass != truth_pass:
        return 0.0
    elif not truth_errors:
        return 1.0 if not pred_errors else 0.0
    
    # Handle cases where both fail with errors
    N = len(truth_errors)
    if N == 0:
        return 0.0
    
    total_score = 0.0
    pred_lines = {e.line for e in pred_errors}
    
    for truth_error in truth_errors:
        # Check if there's a matching prediction for this line
        if truth_error.line in pred_lines:
            # Find the corresponding predicted error
            pred_error = next(e for e in pred_errors 
                            if e.line == truth_error.line)
            # Calculate Jaccard similarity for the error messages
            similarity = jaccard_similarity2(truth_error.message, 
                                         pred_error.message)
            total_score += similarity
    
    return total_score / N





model_mapping = {
    # Llama 3.1 models
    "/home/test/test03/models/Meta-Llama-3.1-8B-Instruct": "LLaMA-3.1-8B-Instruct",
    "/home/test/test03/models/Meta-Llama-3.1-70B-Instruct": "LLaMA-3.1-70B-Instruct",

    # Qwen 2.5 base models
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen-2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen-2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen-2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen-2.5-7B-Instruct",
    "/home/test/test03/models/Qwen2.5-14B-Instruct": "Qwen-2.5-14B-Instruct",
    "/home/test/test03/models/Qwen2.5-32B-Instruct": "Qwen-2.5-32B-Instruct",
    "/home/test/test03/models/Qwen2.5-72B-Instruct": "Qwen-2.5-72B-Instruct",

    # Qwen 2.5 Coder models
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": "Qwen-2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "Qwen-2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct": "Qwen-2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen-2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct": "Qwen-2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen-2.5-Coder-32B-Instruct"
}


def parse_numerical_output(output_str):
    """Parse numerical output in various formats"""
    try:
        # Remove any whitespace and newlines
        output_str = output_str.replace('\n', ' ').strip()
        
        # Handle comma-separated scientific notation
        if ',' in output_str and ('e+' in output_str or 'e-' in output_str):
            numbers = [x.strip() for x in output_str.split(',')]
            return [float(x) for x in numbers]
            
        # Handle numpy array with scientific notation
        elif "e+" in output_str or "e-" in output_str:
            cleaned = output_str.replace("'", "").replace("[", "").replace("]", "")
            numbers = [x for x in cleaned.split() if x and x != ' ']
            return [float(x) for x in numbers]
            
        # Handle numpy-style matrix output
        elif '[[' in output_str and ']]' in output_str:
            cleaned = output_str.replace('[', '').replace(']', '')
            numbers = [x for x in cleaned.split() if x and x != ' ']
            return [float(x.rstrip('.')) for x in numbers]
        
        # Handle regular array format [1,2,3]
        elif output_str.startswith('[') and output_str.endswith(']'):
            cleaned = output_str.strip('[]')
            if ',' in cleaned:
                numbers = [x.strip() for x in cleaned.split(',') if x.strip()]
            else:
                numbers = [x.strip() for x in cleaned.split() if x.strip()]
            return [float(x) for x in numbers]
        
        # Handle comma-separated numbers
        elif ',' in output_str:
            numbers = [x.strip() for x in output_str.split(',')]
            return [float(x) for x in numbers]
            
        # Handle single number
        else:
            return [float(output_str)]
            
    except Exception as e:
        raise ValueError(f"Failed to parse numerical output: {str(e)}\nOriginal output: {output_str}")

def extract_output(text):
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Step 1: Locate the "Output:" section and extract its content
    output_match = re.search(r"Output:\s*\n*(.*)", text, re.DOTALL)
    if output_match:
        extracted_text = output_match.group(1).strip()
    else:
        extracted_text = text  # If no "Output:", consider entire text
    
    # Step 2: Remove bold formatting like "**Output:**"
    extracted_text = re.sub(r"\*\*.*?\*\*\s*", "", extracted_text)

    # Step 3: Extract content from the first Markdown code block (``` or ```xxx)
    code_block_match = re.search(r"```(?:\w+)?\n(.*?)\n```", extracted_text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()  # Return content inside the code block
    
    # Step 4: Extract content from inline code (`answer`)
    inline_code_match = re.search(r"`([^`]+)`", extracted_text)
    if inline_code_match:
        return inline_code_match.group(1).strip()  # Return inline code content
    
    # Step 5: If no code blocks, return the first meaningful line
    first_line = extracted_text.split("\n\n")[0].strip()
    # Remove any remaining ** markers
    if first_line.startswith("**"):
        first_line = first_line.replace("**", "").strip()
    if first_line:
        return first_line

    return None  # Return None if no valid output is found


def evaluate_numerical(output, gt, rel_tol=1e-3):
    """Evaluate numerical outputs with relative tolerance"""
    try:
        output_nums = parse_numerical_output(output)
        gt_nums = parse_numerical_output(gt)
        
        # Handle length mismatch
        if len(output_nums) > len(gt_nums):
            output_nums = output_nums[:len(gt_nums)]
        elif len(output_nums) < len(gt_nums):
            output_nums = output_nums + [0] * (len(gt_nums) - len(output_nums))
        
        # Calculate relative errors
        rel_errors = [abs(o - g) / (abs(g) + 1e-10) for o, g in zip(output_nums, gt_nums)]
        max_rel_error = max(rel_errors)
        avg_rel_error = sum(rel_errors) / len(rel_errors)
        
        return {
            "passed": max_rel_error < rel_tol,
            "metrics": {
                "max_relative_error": max_rel_error,
                "avg_relative_error": avg_rel_error,
                "output_length": len(output_nums),
                "expected_length": len(gt_nums)
            }
        }
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "metrics": {}
        }

def evaluate_sorting(output, gt):
    """Evaluate sorting results using rank correlation"""
    try:
        output_nums = parse_numerical_output(output)
        gt_nums = parse_numerical_output(gt)
        rank_correlation = spearmanr(output_nums, gt_nums).correlation
        return {
            "passed": rank_correlation > 0.9,
            "metrics": {
                "rank_correlation": rank_correlation
            }
        }
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "metrics": {}
        }

def jaccard_similarity(str1: str, str2: str) -> float:
    """Calculate Jaccard similarity between two strings"""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def evaluate_exact_match(output, gt):
    """Evaluate exact string match"""
    if output==None or gt==None:
        return {
            "passed": False,
            "metrics": {"exact_match": False, "note": "output or gt is None"}
        }
    else:
        return {
            "passed": output.strip() == gt.strip(),
            "metrics": {
                "exact_match": output.strip() == gt.strip()
            }
        }
    

def evaluate_exact_match(output, gt):
    """Evaluate exact string match"""
    if output==None or gt==None:
        return {
            "passed": False,
            "metrics": {"exact_match": False, "note": "output or gt is None"}
        }
    else:
        return {
            "passed": output.strip() == gt.strip(),
            "metrics": {
                "exact_match": output.strip() == gt.strip()
            }
        }
    
def evaluate_exact_match_3(output, gt, result):
    """Evaluate exact string match"""
    if output==None or gt==None:
        return {
            "passed": False,
            "metrics": {"exact_match": False, "note": "output or gt is None"}
        }
    else:
        return {
            "passed": output.strip() == gt.strip(),
            "metrics": {
                "exact_match": similarity_3(result, gt)
            }
        }

def eval_fl(output, gt, tol=1e-1):
    output = extract_output(output)
    if output==None or gt==None:
        return {
            "passed": False,
            "metrics": {"fl_score": 0.0, "fl_tol": tol, "note": "output or gt is None"}
        }
    fl_score = calculate_fl_score(output, gt)
    return {
        "passed": (1-fl_score) < tol,
        "metrics": {
            "fl_score": fl_score,
            "fl_tol": tol
        }
    }


def evaluate_output(output, gt, fileid):
    """Evaluate output based on file type"""
    output = extract_output(output)
    if output==None or gt==None:
        return {
            "passed": False,
            "metrics": {"exact_match": False, "note": "output or gt is None"}
        }
    # Numerical computation tasks
    if any(x in fileid for x in ['gd_', 'rk_', 'qr_', 'lu_', 'power_method_', 'heat_eq_', 'euler_', 'fft_']):
        return evaluate_numerical(output, gt, rel_tol=1e-2)
        
    # Sorting tasks
    elif any(x in fileid for x in ['sort']):
        return evaluate_sorting(output, gt)
        
    # Monte Carlo estimation
    elif fileid.startswith('mc_'):
        output = output.split(":")[-1].strip()
        gt = gt.split(":")[-1].strip()
        return evaluate_numerical(output, gt, rel_tol=1e-2)  # More relaxed tolerance for MC
        
    # String matching, binary search, hamiltonian cycle, etc.
    else:
        return evaluate_exact_match(output, gt)
def eval_jaccard(output, gt, tol=1e-1):
    if output==None or gt==None:
        return {
            "passed": False,
            "metrics": {"jaccard": 0.0, "jaccard_tol": tol, "note": "output or gt is None"}
        }
    jaccard = jaccard_similarity(output, gt)
    return {
            "passed": (1.0-jaccard) < tol,
            "metrics": {
                "jaccard": jaccard,
                "jaccard_tol": tol,
            }
        }

def evaluate_results(results_file, save_details=True):
    """Evaluate results grouped by category, only process category 4 and 5"""
    # Read JSONL file
    results = []
    with jsonlines.open(results_file) as reader:
        for obj in reader:
            results.append(obj)
    
    # Group results by category
    results_by_category = {}
    for result in results: # for classes 1,2 there is miss-spelling of category, but has been corrected in convert_to_jsonl.py
        category = result.get("custom_category")
        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append(result)
    
    # Only process category 4 and 5
    total_results = []
    
    base_models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct", 
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct", 
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        
        "/home/test/test03/models/Meta-Llama-3.1-8B-Instruct",
        "/home/test/test03/models/Qwen2.5-14B-Instruct",
        "/home/test/test03/models/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "/home/test/test03/models/Meta-Llama-3.1-70B-Instruct",
        "/home/test/test03/models/Qwen2.5-72B-Instruct"
    ]
    
    suffixes = ['sys_0shot', 'sys_3shot', 'sys_cot']
    
    # Create combinations of models and suffixes
    output_keys = [
        f"{model}{suffix}"
        for model in base_models
        for suffix in suffixes
    ]
    
    for model, output_key in zip(base_models, output_keys):
        print(f"###Evaluating model: {model_mapping[model]}###")
        
        for category in [1,2]:
            if category not in results_by_category:
                continue
                
            category_results = results_by_category[category]
            total = len(category_results)
            passed = 0
            detailed_results = []
            
            print(f"\nProcessing Category {category}:")
            print(f"Total samples: {total}")
            
            for result in category_results:
                if output_key not in result:
                    print(f"Skipping result because {output_key} not in result")
                    continue
                output = result[output_key]
                gt = result["gt"]
                
                output = extract_output(output)
                eval_result=evaluate_exact_match(output, gt)
                # Convert numpy types to native Python types
                metrics = {}
                for k, v in eval_result["metrics"].items():
                    if isinstance(v, (np.integer, np.floating)):
                        metrics[k] = float(v)
                    elif isinstance(v, (np.bool_)):
                        metrics[k] = bool(v)
                    else:
                        metrics[k] = v
                
                # Store detailed result
                detailed_result = {
                    "category": category,
                    "output": output,
                    "ground_truth": gt,
                    "evaluation": {
                        "passed": bool(eval_result["passed"]),
                        "metrics": metrics,
                        "error": eval_result.get("error", None)
                    },
                    "passed": bool(eval_result["passed"])
                }
                detailed_results.append(detailed_result)
                
                # Update statistics
                if eval_result["passed"]:
                    passed += 1
            
            # Print category results
            print(f"\nCategory {category} Results:")
            print(f"Total: {total}")
            print(f"Passed: {passed}")
            print(f"Accuracy: {passed/total*100:.2f}%")

            save_overall_results(model_mapping[model], output_key, category, total, passed)
            total_results.extend(detailed_results)
        for category in [3]:
            if category not in results_by_category:
                continue
                
            category_results = results_by_category[category]
            total = len(category_results)
            passed = 0
            detailed_results = []
            
            print(f"\nProcessing Category {category}:")
            print(f"Total samples: {total}")
            
            for result in category_results:
                if output_key not in result:
                    print(f"Skipping result because {output_key} not in result")
                    continue
                output = result[output_key]
                gt = result["gt"]
                
                output = extract_output(output)
                eval_result=evaluate_exact_match_3(output, gt, result)
                # Convert numpy types to native Python types
                metrics = {}
                for k, v in eval_result["metrics"].items():
                    if isinstance(v, (np.integer, np.floating)):
                        metrics[k] = float(v)
                    elif isinstance(v, (np.bool_)):
                        metrics[k] = bool(v)
                    else:
                        metrics[k] = v
                
                # Store detailed result
                detailed_result = {
                    "category": category,
                    "output": output,
                    "ground_truth": gt,
                    "evaluation": {
                        "passed": bool(eval_result["passed"]),
                        "metrics": metrics,
                        "error": eval_result.get("error", None)
                    },
                    "passed": bool(eval_result["passed"])
                }
                detailed_results.append(detailed_result)
                
                # Update statistics
                if eval_result["passed"]:
                    passed += 1
            
            # Print category results
            print(f"\nCategory {category} Results:")
            print(f"Total: {total}")
            print(f"Passed: {passed}")
            print(f"Accuracy: {passed/total*100:.2f}%")

            save_overall_results(model_mapping[model], output_key, category, total, passed)
            total_results.extend(detailed_results)
        for category in [4,5]:
            if category not in results_by_category:
                continue
                
            category_results = results_by_category[category]
            total = len(category_results)
            passed = 0
            metrics_by_type = {}
            detailed_results = []
            
            print(f"\nProcessing Category {category}:")
            print(f"Total samples: {total}")
            
            for result in category_results:
                if "custom_features" not in result or "fid" not in result["custom_features"]:
                    print(f"Skipping result because it doesn't have custom_features or fid")
                    continue
                    
                fileid = result["custom_features"]["fid"]
                if output_key not in result:
                    print(f"Skipping result because {output_key} not in result")
                    continue
                output = result[output_key]
                gt = result["gt"]
                
                eval_result = evaluate_output(output, gt, fileid)
                
                # Convert numpy types to native Python types
                metrics = {}
                for k, v in eval_result["metrics"].items():
                    if isinstance(v, (np.integer, np.floating)):
                        metrics[k] = float(v)
                    elif isinstance(v, (np.bool_)):
                        metrics[k] = bool(v)
                    else:
                        metrics[k] = v
                
                # Store detailed result
                detailed_result = {
                    "category": category,
                    "fileid": fileid,
                    "output": output,
                    "ground_truth": gt,
                    "evaluation": {
                        "passed": bool(eval_result["passed"]),
                        "metrics": metrics,
                        "error": eval_result.get("error", None)
                    },
                    "passed": bool(eval_result["passed"])
                }
                detailed_results.append(detailed_result)
                
                # Update statistics
                if eval_result["passed"]:
                    passed += 1
                    
                # Track metrics by file type
                file_type = fileid.split('_')[0] if '_' in fileid else fileid.replace('.py', '')
                if file_type not in metrics_by_type:
                    metrics_by_type[file_type] = {"total": 0, "passed": 0}
                metrics_by_type[file_type]["total"] += 1
                if eval_result["passed"]:
                    metrics_by_type[file_type]["passed"] += 1
            
            # Print category results
            print(f"\nCategory {category} Results:")
            print(f"Total: {total}")
            print(f"Passed: {passed}")
            print(f"Accuracy: {passed/total*100:.2f}%")
            
            print(f"\nCategory {category} Results by type:")
            for file_type, stats in metrics_by_type.items():
                accuracy = stats["passed"] / stats["total"] * 100
                print(f"{file_type}: {stats['passed']}/{stats['total']} ({accuracy:.2f}%)")
            save_overall_results(model_mapping[model], output_key, category, total, passed)
            total_results.extend(detailed_results)

        for category in [6,7]:
            if category not in results_by_category:
                continue
                
            category_results = results_by_category[category]
            total = len(category_results)
            passed = 0
            detailed_results = []
            
            print(f"\nProcessing Category {category}:")
            print(f"Total samples: {total}")
            
            for result in category_results:
                if output_key not in result:
                    print(f"Skipping result because {output_key} not in result")
                    continue
                output = result[output_key]
                gt = result["gt"]
                
                output = extract_output(output)
                eval_result=eval_jaccard(output, gt)
                # Convert numpy types to native Python types
                metrics = {}
                for k, v in eval_result["metrics"].items():
                    if isinstance(v, (np.integer, np.floating)):
                        metrics[k] = float(v)
                    elif isinstance(v, (np.bool_)):
                        metrics[k] = bool(v)
                    else:
                        metrics[k] = v
                
                # Store detailed result
                detailed_result = {
                    "category": category,
                    "output": output,
                    "ground_truth": gt,
                    "evaluation": {
                        "passed": bool(eval_result["passed"]),
                        "metrics": metrics,
                        "error": eval_result.get("error", None)
                    },
                    "passed": bool(eval_result["passed"])
                }
                detailed_results.append(detailed_result)
                
                # Update statistics
                if eval_result["passed"]:
                    passed += 1
            
            # Print category results
            print(f"\nCategory {category} Results:")
            print(f"Total: {total}")
            print(f"Passed: {passed}")
            print(f"Accuracy: {passed/total*100:.2f}%")

            save_overall_results(model_mapping[model], output_key, category, total, passed)
            total_results.extend(detailed_results)
        for category in [8]:
            if category not in results_by_category:
                continue
                
            category_results = results_by_category[category]
            total = len(category_results)
            passed = 0
            detailed_results = []
            
            print(f"\nProcessing Category {category}:")
            print(f"Total samples: {total}")
            
            for result in category_results:
                if output_key not in result:
                    print(f"Skipping result because {output_key} not in result")
                    continue
                output = result[output_key]
                gt = result["gt"]
                eval_result=eval_fl(output, gt)
                # Convert numpy types to native Python types
                metrics = {}
                for k, v in eval_result["metrics"].items():
                    if isinstance(v, (np.integer, np.floating)):
                        metrics[k] = float(v)
                    elif isinstance(v, (np.bool_)):
                        metrics[k] = bool(v)
                    else:
                        metrics[k] = v
                
                # Store detailed result
                detailed_result = {
                    "category": category,
                    "output": output,
                    "ground_truth": gt,
                    "evaluation": {
                        "passed": bool(eval_result["passed"]),
                        "metrics": metrics,
                        "error": eval_result.get("error", None)
                    },
                    "passed": bool(eval_result["passed"])
                }
                detailed_results.append(detailed_result)
                
                # Update statistics
                if eval_result["passed"]:
                    passed += 1
            
            # Print category results
            print(f"\nCategory {category} Results:")
            print(f"Total: {total}")
            print(f"Passed: {passed}")
            print(f"Accuracy: {passed/total*100:.2f}%")
            save_overall_results(model_mapping[model], output_key, category, total, passed)
            
            total_results.extend(detailed_results)
        # Save detailed results to file
        if save_details:
            output_file = results_file.replace('.jsonl', '_evaluated.jsonl')
            with jsonlines.open(output_file, mode='w') as writer:
                writer.write_all(total_results)
            print(f"\nDetailed results saved to: {output_file}")
    
    return total_results


def edit_distance(s1, s2):
    def dp(i, j):
        # base case
        if i == -1: return j + 1
        if j == -1: return i + 1
        
        if s1[i] == s2[j]:
            return dp(i-1, j-1)  # 跳过
        else:
            return min(
                dp(i, j-1) + 1,   # 插入
                dp(i-1, j) + 1,   # 删除
                dp(i-1, j-1) + 1  # 替换
            )
    return dp(len(s1)-1, len(s2)-1)
def match(str1, str2)->int:
    if(str1 == str2):return 1
    return 0

from difflib import SequenceMatcher
def explain_sequence_matcher(str1, str2):
    # 创建SequenceMatcher对象
    matcher = SequenceMatcher(None, str1, str2)
    matching_blocks = matcher.get_matching_blocks()
    
    for block in matching_blocks:
        i, j, size = block
    similarity = matcher.ratio()
    return similarity
def similarity_3(entry: dict, output):
    if(entry["custom_features"]["repo"] == "TUOJ"):
        return match(entry["gt"], output)
    elif(entry["custom_features"]["repo"] == "PlotNeuralNet"):
        return explain_sequence_matcher(entry["gt"], output)
    else:
        return edit_distance(entry["gt"], output) / (len(entry["gt"]) + len(output)  )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True,
                      help='Path to results file')
    parser.add_argument('--no-save', action='store_true',
                      help='Do not save detailed results')
    args = parser.parse_args()
    
    evaluate_results(args.results, save_details=not args.no_save)
