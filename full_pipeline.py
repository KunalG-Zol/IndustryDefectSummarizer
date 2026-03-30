import os
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

warnings.filterwarnings('ignore')

model_path = r"D:\HFModels\NousResearch_Llama3-8B-Instruct"
lora_path = r".\defect_lora_weights"
vector_db_path = r".\faiss_qc_index"

#Config for 4 bit quantization
bnb_conf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_conf,
    dtype=torch.bfloat16,
    device_map='auto')
#Loading the fine tuned lora adapters
model = PeftModel.from_pretrained(base_model, lora_path)
#Loading the embedding model
embedding_model = HuggingFaceBgeEmbeddings(
    model_name='BAAI/bge-small-en-v1.5',
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
#Loading the vector db
vector_db = FAISS.load_local(vector_db_path,
                             embedding_model,
                             allow_dangerous_deserialization=True
                             )

raw_report = """
DURING REFUELING OF THE AIRCRAFT, THE FUELER NOTICED A LEAK COMING FROM THE RIGHT WING VENT. 
THE CAPTAIN WAS NOTIFIED AND THE REFUELING WAS STOPPED IMMEDIATELY. MAINTENANCE WAS CALLED 
TO THE GATE. UPON INSPECTION, IT WAS FOUND THAT THE HIGH LEVEL SHUTOFF VALVE HAD FAILED TO 
CLOSE, CAUSING FUEL TO ENTER THE VENT SYSTEM. THE VALVE WAS REPLACED AND TESTED SATISFACTORY.
"""

prompt_step_1 = f"""<|start_header_id|>system<|end_header_id|>
You are a technical aviation safety analyst. Summarize the following defect report into a concise, professional synopsis.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Raw Report: {raw_report}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Synopsis: """

inputs_1 = tokenizer(prompt_step_1, return_tensors="pt").to("cuda")

with torch.no_grad():
    ft_output = model.generate(**inputs_1, max_new_tokens=100)

clean_synopsis = tokenizer.decode(ft_output[0], skip_special_tokens=True).split("Synopsis: ")[-1].strip()
print("\n--- 1. STANDARDIZED SYNOPSIS ---")
print(clean_synopsis)

retrieved_docs = vector_db.similarity_search(clean_synopsis, k=3)
context = "\n".join([doc.page_content for doc in retrieved_docs])

print("\n--- 2. RETRIEVED HANDBOOK CONTEXT ---")
print(context)

prompt_step_2 = f"""<|start_header_id|>system<|end_header_id|>
You are an aviation maintenance expert. Provide a step-by-step repair plan using ONLY the provided handbook context. Do not add outside information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Defect: {clean_synopsis}

Handbook Context: 
{context}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Repair Plan: """

inputs_2 = tokenizer(prompt_step_2, return_tensors="pt").to("cuda")

with model.disable_adapter():
    with torch.no_grad():
        final_output = model.generate(**inputs_2, max_new_tokens=600)

print("\n--- 3. FINAL REPAIR PLAN ---")
print(tokenizer.decode(final_output[0], skip_special_tokens=True).split("Repair Plan: ")[-1].strip())


print("\n" + "="*50)
print("--- BASE MODEL PIPELINE (RAW INPUT, NO ADAPTERS) ---")

base_retrieved_docs = vector_db.similarity_search(raw_report, k=3)
base_context = "\n".join([doc.page_content for doc in base_retrieved_docs])

prompt_base = f"""<|start_header_id|>system<|end_header_id|>
You are an aviation maintenance expert. Provide a step-by-step repair plan using ONLY the provided handbook context. Do not add outside information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Defect: {raw_report}

Handbook Context: 
{base_context}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Repair Plan: """

inputs_base = tokenizer(prompt_base, return_tensors="pt").to("cuda")

with model.disable_adapter():
    with torch.no_grad():
        base_output = model.generate(**inputs_base, max_new_tokens=600)

print("\n--- BASE RETRIEVED CONTEXT ---")
print(base_context)
print("\n--- BASE REPAIR PLAN ---")
print(tokenizer.decode(base_output[0], skip_special_tokens=True).split("Repair Plan: ")[-1].strip())