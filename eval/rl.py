import json
import re 

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

def extract_sudoku(text):
    # Remove leading/trailing whitespace
    text = text.strip()

    pattern = r"------- 0 -------"
    matches = list(re.finditer(pattern, text))

    if len(matches) > 0:
        ans = text[matches[-1].start()-18*9-2:matches[-1].end()]
        return ans
    else:
        return None

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

def rl_eval(entry: dict, key):
    try:
    
        output = extract_output(entry[key])
        # if(entry["custom_features"]["task"] == 'sudoku'):
        #     print(entry["custom_features"]["task"], key, output)
        if entry["custom_features"]["task"] == "24_game":
            return explain_sequence_matcher(entry["gt"], output)
        
        elif entry["custom_features"]["task"] == "plotneurnalnet":
            return explain_sequence_matcher(entry["gt"], output)
        elif entry["custom_features"]["task"] == "sudoku":
            str1, str2 = '', ''
            ans1, ans2 = 0, 0
            if extract_output(entry[key]):
                str1 = extract_output(entry[key])
            if extract_sudoku(entry[key]):
                str2 = extract_output(extract_sudoku(entry[key]))
            ans1 = explain_sequence_matcher(entry["gt"], str1)
            ans2 = explain_sequence_matcher(entry["gt"], str2)
            return max(ans1, ans2)
            
        
        elif entry["custom_features"]["task"] == "TUOJ,cow":
            return match(entry["gt"], output)
        elif entry["custom_features"]["task"] == "TUOJ,car":
            return match(entry["gt"], output)
        
        else:
            print("error", entry["custom_features"]["task"])
    except:
        return 0    

def statistic(data, key):
    total = 0
    correct = 0
    sudoku_list = []
    _24game_list = []
    plotneurnalnet_list = []
    thuoj_list_cow = []
    thuoj_list_car = []
    
    for entry in data:
        # print(total)
        total += 1
        
        if(entry["custom_features"]["repo"] == "PlotNeuralNet"):
            entry["custom_features"]["language"] = "Python"
            entry["custom_features"]["task"] = "plotneurnalnet"
            plotneurnalnet_list.append(total)
            continue
        elif(entry["custom_features"]["run_instruction"] == "python 24_game.py"):
            _24game_list.append(total)
            entry["custom_features"]["task"] = "24_game"
            continue
        elif(entry["custom_features"]["repo"] == "https://github.com/psychopurp/sudoku-player"):
            sudoku_list.append(total)
            continue
        else:
            if entry["custom_features"]["run_instruction"] == "make":
                thuoj_list_cow.append(total)
                entry["custom_features"]["task"] = "TUOJ,cow"
            else:
                thuoj_list_car.append(total)
                entry["custom_features"]["task"] = "TUOJ,car"
            entry["custom_features"]["language"] = "CPP"
            continue

    score = correct / total
    print("sudoku:",sudoku_list)
    print("24game:",_24game_list)
    print("plotneurnalnet:",plotneurnalnet_list)
    print("thuoj_cow:",thuoj_list_cow)
    print("thuoj_car:",thuoj_list_car)
    return score, data