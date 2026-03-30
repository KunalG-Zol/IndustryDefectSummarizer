import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

local_path = r"D:\HFModels\NousResearch_Llama3-8B-Instruct"
adapter_path = "./defect_lora_weights/"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(local_path)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    local_path,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map="auto"
)

raw_report = """
DURING REFUELING OF THE AIRCRAFT, THE FUELER NOTICED A LEAK COMING FROM THE RIGHT WING VENT. 
THE CAPTAIN WAS NOTIFIED AND THE REFUELING WAS STOPPED IMMEDIATELY. MAINTENANCE WAS CALLED 
TO THE GATE. UPON INSPECTION, IT WAS FOUND THAT THE HIGH LEVEL SHUTOFF VALVE HAD FAILED TO 
CLOSE, CAUSING FUEL TO ENTER THE VENT SYSTEM. THE VALVE WAS REPLACED AND TESTED SATISFACTORY.
"""

prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a technical aviation safety analyst. Summarize the following defect report into a concise, professional synopsis.
<|start_header_id|>user<|end_header_id|>
Raw Report: {raw_report}
<|start_header_id|>assistant<|end_header_id|>
Synopsis: """

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("--- GENERATING WITH BASE MODEL ---")
with torch.no_grad():
    base_output = base_model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(base_output[0], skip_special_tokens=True).split("Synopsis: ")[-1])

print("\n" + "="*50 + "\n")

print("--- LOADING ADAPTERS ---")
ft_model = PeftModel.from_pretrained(base_model, adapter_path)

print("--- GENERATING WITH FINE-TUNED MODEL ---")
with torch.no_grad():
    ft_output = ft_model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(ft_output[0], skip_special_tokens=True).split("Synopsis: ")[-1])