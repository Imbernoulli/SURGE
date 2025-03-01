import json
import jsonlines
from scipy.stats import spearmanr
import numpy as np
import re
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

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
    try:
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
    except:
        return None


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
        score=np.clip(1-avg_rel_error, 0, 1) # 1-avg_rel_error is the score， clip to 0-1
        return score
    except Exception as e:
        return 0

def evaluate_sorting(output, gt):
    """Evaluate sorting results using rank correlation"""
    try:
        output_nums = parse_numerical_output(output)
        gt_nums = parse_numerical_output(gt)
        rank_correlation = spearmanr(output_nums, gt_nums).correlation
        score=np.clip(rank_correlation, 0, 1)
        return score
    except Exception as e:
        return 0

def evaluate_output(output, gt, fileid):
    """Evaluate output based on file type"""
    
    if output==None or gt==None:
        return 0
    output = extract_output(output)
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
        return output == gt