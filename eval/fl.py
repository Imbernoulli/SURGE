import re
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Error:
    line: int
    message: str

def extract_code(content: str) -> str:
    """
    从 Markdown 代码块或 JSON 代码块中提取内容，并去除 JSON 末尾的多余逗号。
    
    :param content: 包含 ```xxx``` 或 ```json\nxxx``` 形式的字符串
    :return: 处理后的 JSON 字符串
    """
    match = re.search(r'```(?:json\n)?(.*?)```', content, re.DOTALL)
    json_str = match.group(1).strip() if match else content.strip()
    
    # 去除 JSON 末尾的多余逗号
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
    
    # 处理双重转义的问题
    json_str = json_str.replace('\\"', '"').replace('\\\\', '\\')
    
    return json_str

def parse_json_result(json_str: str) -> Tuple[bool, List[Error]]:
    """Parse JSON result string into pass status and list of errors"""
    result = json.loads(json_str)
    errors = []
    for error in result.get("errors", []):
        pos = error["pos"]
        errors.append(Error(
            line=pos["line"],
            message=error["data"]
        ))
    return result.get("pass", False), errors

def jaccard_similarity(str1: str, str2: str) -> float:
    try:
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    except:
        return 0

def fl_eval(pred_json: str, truth_json: str) -> float:
    try:
        pred_pass, pred_errors = parse_json_result((pred_json))
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
                similarity = jaccard_similarity(truth_error.message, 
                                            pred_error.message)
                total_score += similarity
        
        return total_score / N
    except Exception as e:
        print(e)

gt = "{\"errors\": [{\"severity\": \"error\", \"pos\": {\"line\": 35, \"column\": 2}, \"endPos\": {\"line\": 35, \"column\": 7}, \"data\": \"omega could not prove the goal:\\na possible counterexample may satisfy the constraints\\n  b \\u2265 2\\n  2*a + b \\u2264 179\\n  a + b \\u2264 178\\n  2 \\u2264 a \\u2264 4\\n  a + 2*b \\u2265 181\\n  a - b \\u2264 -1\\nwhere\\n a := \\u2191q\\n b := \\u2191\\u03b8\"}], \"pass\": false, \"complete\": false}"

pred = "Thought: The provided Lean4 code attempts to prove that in an isosceles triangle with two distinct prime angles \\( p \\) and \\( q \\) (where \\( p > q \\)), the angle \\( q \\) must be at least 5. The proof uses properties of prime numbers and the sum of angles in a triangle. However, the use of `omega` at the end suggests that the proof is relying on a tactic that may not be applicable in this context, as `omega` is typically used for linear arithmetic and may not handle the logical structure of the proof correctly. This could lead to a failure in compilation due to the inability to resolve the proof obligations.\n\nOutput:\n```json\n{\n    \"errors\": [\n        {\n            \"severity\": \"error\",\n            \"pos\": {\n                \"line\": 25,\n                \"column\": 2\n            },\n            \"endPos\": {\n                \"line\": 25,\n                \"column\": 6\n            },\n            \"data\": \"invalid tactic application\"\n        }\n    ],\n    \"pass\": false,\n    \"complete\": false\n}\n```"