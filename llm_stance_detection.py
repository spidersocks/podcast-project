import os
import re
import gc
import glob
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import logging
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# defining file and directory paths
INPUT_CSV = "data/chunks_for_stance_detection.csv"
OUTPUT_DIR = "/home/ubuntu/results_batches/"
FINAL_PARQUET = "/home/ubuntu/chunks_with_stances_final.parquet"

# model and endpoint settings
MODEL_NAME = "./Mistral-7B-Instruct-v0.2-GPTQ"
VLLM_BASE_URL = "http://localhost:8000/v1"
OPENAI_API_KEY = "EMPTY"  # local vLLM endpoint

# processing parameters
BATCH_SIZE = 32
MAX_WORKERS = 8
USE_EXAMPLE = False
EXAMPLE_DICT = None

# set up logging
logging.basicConfig(filename="stance_detection.log", filemode="a",
                    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
random.seed(42)
np.random.seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
client = OpenAI(base_url=VLLM_BASE_URL, api_key=OPENAI_API_KEY)

def create_cos_prompt(text, topic, source_name, title, example=None):
    instruction = (
        "You are an expert in stance detection. "
        "For the following text, determine the stance towards the given topic as FAVOR, AGAINST, or NONE. "
        "In your analysis, consider all of the following: "
        "1) the context and background of the text, "
        "2) the main viewpoint or intent, "
        "3) the emotional tone and language, "
        "4) compare evidence for each possible stance (FAVOR, AGAINST, NONE), "
        "and 5) make a logical, consistent decision. "
        "You may summarize your reasoning in a few sentences, but you do not need to write a paragraph for each step. "
        "At the end, output your answer on a new line in the format: Final Stance: [STANCE] (where [STANCE] is FAVOR, AGAINST, or NONE)."
    )
    def task_prompt(source_name, title, text, topic):
        return (
            f"{instruction}\n\n"
            f"Source: {source_name}\n"
            f"Title: {title}\n"
            f"Text: {text}\n"
            f"Topic: {topic}\n"
            f"Begin."
        )
    if example is not None:
        messages = [
            {"role": "user", "content": task_prompt(
                example['source_name'], example['title'], example['text'], example['topic'])},
            {"role": "assistant", "content": example['answer']},
            {"role": "user", "content": task_prompt(source_name, title, text, topic)}
        ]
    else:
        messages = [
            {"role": "user", "content": task_prompt(source_name, title, text, topic)}
        ]
    return messages

def extract_final_stance(output):
    match = re.search(r"Final Stance\s*:\s*\[?\s*(FAVOR|AGAINST|NONE)\s*\]?", output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match2 = re.search(r'(FAVOR|AGAINST|NONE)', output, re.IGNORECASE)
    if match2:
        return match2.group(1).upper()
    return "ERROR"

def generate_stance_requests(df):
    """
    Generate a flat list of (row_idx, topic_name) for all stance requests.
    """
    stance_requests = []
    topic_cols = [col for col in df.columns if col.startswith("topic_")]
    for row_idx, row in df.iterrows():
        for col in topic_cols:
            if row.get(col, False):
                topic_name = col.replace("topic_", "").replace("_", " ")
                stance_requests.append((row_idx, topic_name))
    return stance_requests

def vllm_generate_single(messages, *,
                         model_name=MODEL_NAME,
                         max_tokens=128,
                         temperature=0.0,
                         max_retries=2):
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,   
                temperature=temperature,
                stream=False,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logging.error(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                return f"ERROR: {e}"

def vllm_parallel_generate(message_batches, max_workers=MAX_WORKERS, **kwargs):
    results = [None] * len(message_batches)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(vllm_generate_single, msgs, **kwargs): i
            for i, msgs in enumerate(message_batches)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = f"ERROR: {e}"
    return results

def run_production_stance_detection_json(
    df, batch_size=BATCH_SIZE, use_example=USE_EXAMPLE, output_path=OUTPUT_DIR, example_dict=EXAMPLE_DICT
):
    """
    Process stance requests in batches of `batch_size` (not document rows).
    Each stance request is (row_idx, topic_name).
    Already completed batches are skipped.
    """
    import os
    import glob
    import json
    from tqdm import tqdm

    os.makedirs(output_path, exist_ok=True)
    stance_requests = generate_stance_requests(df)

    existing_batches = set()
    for f in glob.glob(os.path.join(output_path, "batch_*.jsonl")):
        try:
            n = int(os.path.basename(f).replace("batch_", "").replace(".jsonl", ""))
            existing_batches.add(n)
        except Exception:
            pass

    total_batches = (len(stance_requests) + batch_size - 1) // batch_size

    for batch_num in tqdm(range(1, total_batches + 1)):
        batch_file = os.path.join(output_path, f"batch_{batch_num}.jsonl")
        if batch_num in existing_batches:
            print(f"Skipping {batch_file} (already exists)")
            continue

        batch_slice = stance_requests[(batch_num-1)*batch_size : batch_num*batch_size]
        batch_messages = []
        batch_meta = []

        for row_idx, topic_name in batch_slice:
            row = df.iloc[row_idx]
            messages = create_cos_prompt(
                row["text"],
                topic_name,
                row["source_name"],
                row["title"],
                example=example_dict if use_example else None,
            )
            batch_messages.append(messages)
            batch_meta.append({"index": row_idx, "topic": topic_name})

        batch_results = vllm_parallel_generate(batch_messages, max_workers=MAX_WORKERS)
        batch_output = []
        for meta_item, output in zip(batch_meta, batch_results):
            stance = extract_final_stance(output)
            batch_output.append({
                "index": meta_item["index"],
                "topic": meta_item["topic"],
                "stance": stance,
                "reasoning": output
            })

        with open(batch_file, "w") as f:
            for entry in batch_output:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Saved {batch_file} ({len(batch_output)} items)")
    print("All batches processed and saved as JSONL.")

def assemble_final_results(original_df, results_path=OUTPUT_DIR):
    all_files = sorted(glob.glob(os.path.join(results_path, "batch_*.jsonl")))
    records = []
    for file in all_files:
        with open(file, "r") as f:
            for line in f:
                records.append(json.loads(line))
    if not records:
        raise ValueError("No results found in batch JSONL files!")

    results_df = pd.DataFrame(records)
    stance_df = results_df.pivot(
        index="index",
        columns="topic",
        values="stance"
    )
    stance_df.columns = [f"stance_{c.replace(' ', '_')}" for c in stance_df.columns]
    reasoning_df = results_df.pivot(
        index="index",
        columns="topic",
        values="reasoning"
    )
    reasoning_df.columns = [f"reasoning_{c.replace(' ', '_')}" for c in reasoning_df.columns]

    final_df = original_df.copy()
    final_df = final_df.join(stance_df, how="left")
    final_df = final_df.join(reasoning_df, how="left")
    return final_df

def main():
    print("Loading input CSV...")
    chunks_w_topic_labels = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(chunks_w_topic_labels)} rows.")

    print("Beginning batch stance detection...")
    run_production_stance_detection_json(
        chunks_w_topic_labels,
        batch_size=BATCH_SIZE,
        use_example=USE_EXAMPLE,
        output_path=OUTPUT_DIR,
        example_dict=EXAMPLE_DICT
    )

    print("Batch stance detection complete. Assembling final DataFrame...")
    final_df = assemble_final_results(chunks_w_topic_labels, OUTPUT_DIR)
    final_df.to_parquet(FINAL_PARQUET)
    print(f"Stance detection complete. Results saved to '{FINAL_PARQUET}'.")

if __name__ == "__main__":
    main()