import pandas as pd
import json

df = pd.read_csv("asrs.csv", low_memory=False)

df = df[['Report 1.Narrative', 'Report 1.Synopsis', 'Assessments.Primary Problem']]
df = df.rename(columns={
    'Report 1.Narrative': 'Narrative',
    'Report 1.Synopsis': 'Synopsis',
    'Assessments.Primary Problem': 'Primary Problem'
})

df = df.dropna(subset=['Narrative', 'Synopsis', 'Primary Problem'])

target_problems = [
    'Aircraft',
    'Equipment / Tooling',
    'Incorrect / Not Installed / Unavailable Part',
    'Software and Automation',
    'MEL'
]

df_defects = df[df['Primary Problem'].isin(target_problems)]

print(f"Filtered down to {len(df_defects)} pure mechanical/defect reports.")

formatted_data = []
system_prompt = "You are an aviation safety analyst. Summarize the following messy mechanical defect narrative into a concise synopsis."

for index, row in df_defects.iterrows():
    messy_input = row['Narrative']
    clean_target = row['Synopsis']

    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRaw Report: {messy_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{clean_target}<|eot_id|>"

    formatted_data.append({"text": prompt})

with open("defects_train.jsonl", "w") as f:
    for item in formatted_data:
        f.write(json.dumps(item) + "\n")