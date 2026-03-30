# Industry Defect Summarizer & Hallucination-Resistant RAG Pipeline

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline built to process noisy, unstructured industrial defect reports and generate highly accurate, step-by-step repair plans. 

This project solves a critical flaw in standard RAG architectures—Semantic Retrieval Hallucination—by introducing a **Query Transformation Layer** powered by a custom fine-tuned Large Language Model.

## 🧠 System Architecture

Instead of embedding raw user input directly into a vector database (which often retrieves irrelevant context due to narrative noise), this pipeline utilizes a two-step LLM routing strategy:

1. **Query Standardization (Adapter ON):** A LoRA fine-tuned version of `Llama-3-8B-Instruct` intercepts the raw, conversational defect report and strips all narrative fluff, generating a dense, technical ASRS-style synopsis.
2. **Dense Vector Retrieval:** The standardized synopsis is embedded using `BAAI/bge-small-en-v1.5` and queried against a local, in-memory `FAISS` vector index containing technical maintenance handbooks.
3. **Contextual Generation (Adapter OFF):** Using dynamic adapter toggling, the LoRA weights are disabled in RAM. The retrieved context and the clean synopsis are passed to the articulate Base Llama-3 model to generate a chatty, step-by-step repair plan.

## ⚡ Engineering Constraints & Optimizations

This pipeline is aggressively optimized to run entirely locally on consumer hardware with strict memory limits, specifically targeting **8GB VRAM** constraints (e.g., RTX 4060 environments).

* **4-Bit Quantization:** Utilizes `bitsandbytes` (NF4, double-quantization, `bfloat16` compute dtype) to compress the 16GB Llama-3 model footprint down to ~5.5GB.
* **Dynamic PEFT Toggling:** Uses `with model.disable_adapter():` to switch between the highly specialized summarization model and the conversational base model without loading two separate 8B parameter models into memory, preventing CUDA Out-of-Memory (OOM) crashes.
* **Sub-Millisecond Retrieval:** Implements a local `FAISS` inverted index paired with `BGE-small-v1.5` (utilizing normalized cosine similarity) to keep total retrieval latency under 50ms without relying on external Docker containers or cloud databases.

## 🛠️ Tech Stack

* **Core LLM:** NousResearch/Meta-Llama-3-8B-Instruct
* **Fine-Tuning:** QLoRA (Rank 16, Alpha 32), PEFT
* **Embeddings:** BAAI/bge-small-en-v1.5
* **Vector Store:** FAISS (via LangChain)
* **Compute:** PyTorch (CUDA 13.0 optimized)

## 🚀 Quick Start

### 1. Environment Setup
Ensure you have a CUDA-compatible environment (Windows 11 / Linux) with at least 8GB of VRAM.

```bash
# Install PyTorch for CUDA 13.0
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)

# Install pipeline dependencies
pip install transformers peft bitsandbytes accelerate langchain-community sentence-transformers faiss-cpu
