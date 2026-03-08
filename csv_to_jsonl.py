import pandas as pd
import json

df = pd.read_csv("asrs.csv", low_memory=False)
df = df.dropna(subset=['Report 1.Narrative', 'Report 1.Synopsis'])

formatted_data = []
system_prompt = "You are an aviation safety analyst. Summarize the following messy incident narrative into a concise synopsis."

for index, row in df.iterrows():
    messy_input = row['Report 1.Narrative']
    clean_target = row['Report 1.Synopsis']
    
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRaw Report: {messy_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{clean_target}<|eot_id|>"
    
    formatted_data.append({"text": prompt})

with open("aviation_finetune.jsonl", "w") as f:
    for item in formatted_data:
        f.write(json.dumps(item) + "\n")

print(f"Successfully formatted {len(formatted_data)} records")