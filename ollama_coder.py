import requests
import json
import time
import os
from transformers import AutoTokenizer, PreTrainedTokenizer
import pandas as pd
import ast
import re

# The function need to generate a docstring for
function_code = """
from typing import List

def below_zero(operations: List[int]) -> bool:
    \"\"\"
    You're given a list of deposit and withdrawal operations
    on a bank account that starts with zero balance. Your task is to
    detect if at any point the balance of account falls below zero,
    and at that point function should return True.
    \"\"\"
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False
"""

NUM_EPOCHS = 5
# 1. Define the model and the prompt
MODELS_TO_TEST = [
    "mistral:7b-instruct-v0.3",
    "llama3.1:8b-instruct",
    "deepseek-coder:6.7b-instruct",

]

# Standard Ollama tags for quantization: f16 is used for unquantized float16
QUANTIZATION_LEVELS = ["fp16", "q8_0", "q4_0"]



complete_prompt_text = f"""
Please act as an expert Python software engineer. Given the python function below:
{function_code}
I would appreciate it if you could generate a complete and professional Google-style docstring. 
The docstring should not include any extra commentary, strictly limited to include the docstring itself and the original function code. 
CODE ONLY. Use standard Python indentation. Thank you. 
Do not add explanations, notes, or text outside of the code. 
Return only the function code with its docstring, without markdown fences or extra text before or after.
"""


concise_prompt_text = f"""
Generate COMPLETE GOOGLE style docstring for the following Python function:
{function_code}
Output the docstring with the function code. Do not include explanations, notes, or text outside the code. 
Return only the function code with its docstring, without markdown fences or extra text.
"""

ultra_concise_prompt_text = f"""Add GOOGLE style docstring to function:
{function_code}
Output code only, no text."""


PROMPTS = {
    "complete_prompt_text": complete_prompt_text,
    "concise_prompt_text": concise_prompt_text,
    "ultra_concise_prompt_text": ultra_concise_prompt_text
}
# 2. Set up the Ollama API endpoint and payload
# ---------------------------------------------
OLLAMA_ENDPOINT = os.getenv("OLLAMA_BASE_URL", "http://192.168.149.1:11434/api/generate")

MODEL_HF_MAP = {
    "llama3.1:8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "deepseek-coder:6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-base",
    "mistral:7b-instruct-v0.3":"mistralai/Mistral-7B-Instruct-v0.3"
}


def check_code_accuracy(generated_text: str) -> str:
    """
    Valid if:
      - we can extract Python code,
      - there exists a triple-quoted docstring (module- or function-level),
      - the docstring contains 'Args:' or 'Parameters:' and 'Returns:' (case-insensitive).
    Returns "Pass" or "Fail: <reason>".
    """
    if not generated_text:
        return "Fail: Empty output."

    # 1) Extract code (prefers fenced block)
    m = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", generated_text)
    code = (m.group(1) if m else generated_text).strip()
    if not code:
        return "Fail: No code found."

    # 2) Parse with AST
    try:
        mod = ast.parse(code)
    except SyntaxError:
        # Fallback: raw docstring scan
        raw_doc = re.findall(r'("""|\'\'\')([\s\S]*?)\1', code)
        if not raw_doc:
            return "Fail: No triple-quoted docstring found."
        doc_all = " ".join(text for _, text in raw_doc)

        has_args_or_params = re.search(r"(?mi)^\s*(Args|Parameters)\s*:\s*$", doc_all) is not None
        has_returns = re.search(r"(?mi)^\s*Returns\s*:\s*$", doc_all) is not None
        if has_args_or_params and has_returns:
            return "Pass"
        return "Fail: Docstring missing 'Args/Parameters' or 'Returns:'."

    # 3) Collect docstrings (module + any function/class)
    docs = []
    if mod_doc := ast.get_docstring(mod):
        docs.append(mod_doc)
    for node in ast.walk(mod):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if d := ast.get_docstring(node):
                docs.append(d)

    if not docs:
        return "Fail: No triple-quoted docstring found."

    # 4) Check for Args/Parameters and Returns in any docstring
    combined = "\n".join(docs)
    has_args_or_params = re.search(r"(?mi)^\s*(Args|Parameters)\s*:\s*$", combined) is not None
    has_returns = re.search(r"(?mi)^\s*Returns\s*:\s*$", combined) is not None
    if has_args_or_params and has_returns:
        return "Pass"
    return "Fail: Docstring missing 'Args/Parameters' or 'Returns:'."

# --- Helper Function to Calculate Tokens ---
def calculate_tokens(text, _tokenizer):
    if not text or not hasattr(_tokenizer, 'encode'): return 0
    return len(_tokenizer.encode(text))

def generate_with_ollama(
    model_name: str,
    prompt_text: str,
    tokenizer: PreTrainedTokenizer,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    json_format: bool = False,
    seed: int = None,
    stop: list = None
) -> dict:
    """
    Sends a prompt to an Ollama model with advanced options and measures metrics.

    Args:
        model_name (str): The name of the Ollama model.
        prompt_text (str): The full prompt string.
        tokenizer (PreTrainedTokenizer): The tokenizer for calculating token counts.
        temperature (float): Controls randomness. Lower is more deterministic.
        max_tokens (int): The maximum number of tokens to generate.
        json_format (bool): If True, forces the model to output valid JSON.
        seed (int, optional): A seed for reproducible outputs.
        stop (list, optional): A list of strings to stop generation at.

    Returns:
        dict: A dictionary containing the generated text and performance metrics.
    """
    t_in = calculate_tokens(prompt_text, tokenizer)
    print(prompt_text)
    # --- Payload construction inspired by your example ---
    payload = {
        "model": model_name,
        "prompt": prompt_text,  # CLI uses the TEMPLATE with plain prompt by default
        "stream": False,  # one-shot response
        "keep_alive": 0,  # don't retain chat context (matches one-off CLI runs)
        "options": {
            "raw": False,
            "num_predict": int(max_tokens),
            "temperature": float(temperature),
            "top_p": 0.9,
            "top_k": 40,

        },
    }
    if json_format:
        payload["format"] = "json"
    if seed is not None:
        payload["options"]["seed"] = seed
    if stop:
        payload["options"]["stop"] = stop
    # --- End of inspired section ---

    generated_text = "Error: Request failed."
    inference_time = -1.0
    t_out = 0
    result = "N/A"
    try:
        start_time = time.perf_counter()
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=300)
        response.raise_for_status()
        end_time = time.perf_counter()
        generated_text = response.json().get('response', '').strip()
        result = check_code_accuracy(generated_text)
        inference_time = end_time - start_time
        t_out = calculate_tokens(generated_text, tokenizer)

    except requests.exceptions.RequestException as e:
        print(f"  An error occurred for model {model_name}: {e}")
    print(generated_text)
    return {
        "generated_text": generated_text,
        "T_in": t_in,
        "T_out": t_out,
        "T_total": t_in + t_out,
        "inference_time_s": inference_time,
        "result": result
    }

def extract_code_only(text: str) -> str:
    """
    Return only the code inside the first fenced block.
    Supports ```…```, '''…''', and \"\"\"…\"\"\" (optionally with 'python' after the opener).
    Falls back to grabbing from the first 'def ' onward if no fences exist.
    """
    # Try markdown-style fences first
    for fence in ("```", "'''", '"""'):
        # e.g., ```python ... ```  or  ''' ... '''
        pattern = rf"{re.escape(fence)}\s*(?:python)?\s*([\s\S]*?){re.escape(fence)}"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Fallback: no fences — try to slice from first function definition
    i = text.find("def ")
    if i != -1:
        return text[i:].strip()

    # Last resort: return trimmed text
    return text.strip()
if __name__ == "__main__":
    results_list = []
    total_runs = len(MODELS_TO_TEST) * len(PROMPTS) * len(QUANTIZATION_LEVELS) * NUM_EPOCHS
    current_run = 0

    print(f"--- Starting Green Prompting Experiment: {total_runs} runs planned ---")

    # Main experiment loop
    for base_model in MODELS_TO_TEST:
        print(f"\n[Loading Tokenizer for {base_model}]")
        try:
            hf_model_id = MODEL_HF_MAP[base_model]
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        except Exception as e:
            print(f"  Could not load tokenizer for {base_model}. Skipping. Error: {e}")
            current_run += len(PROMPTS) * len(QUANTIZATION_LEVELS) * NUM_EPOCHS
            continue

        for prompt_name, prompt_template in PROMPTS.items():
            for quant in QUANTIZATION_LEVELS:
                full_model_name = f"{base_model}-{quant}"

                for epoch in range(1, NUM_EPOCHS + 1):
                    current_run += 1
                    print(
                        f"--> Running [{current_run}/{total_runs}]: Model={full_model_name}, Prompt='{prompt_name}', Run={epoch}/{NUM_EPOCHS}")

                    # Call the advanced function. A seed is set for determinism check.
                    metrics = generate_with_ollama(
                        model_name=full_model_name,
                        prompt_text=prompt_template,
                        tokenizer=tokenizer,
                        seed=42  # Using a fixed seed for determinism
                    )

                    # Store results along with all experimental variables
                    results_list.append({
                        "Model": base_model,
                        "Quantization": quant,
                        "Prompt": prompt_name,
                        "Epoch": epoch,
                        "Output": metrics["generated_text"],
                        "T_in": metrics["T_in"],
                        "T_out": metrics["T_out"],
                        "T_total": metrics["T_total"],
                        "Inference_Time_s": metrics["inference_time_s"],
                        "Accuracy": metrics["result"],
                        "Hardware": "Local"
                    })

    print("\n--- Experiment Complete ---")

    # Convert the list of results into a pandas DataFrame
    results_df = pd.DataFrame(results_list)

    # Save the final DataFrame to a CSV file
    output_filename = "green_prompt_results_three_models.csv"
    results_df.to_csv(output_filename, index=False)

    print(f"✅ Results for all {total_runs} runs saved to '{output_filename}'")
    print("\n--- Data Sample ---")
    print(results_df.head())