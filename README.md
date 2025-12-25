# Multi-Hop QA with Entailment Verification

This project implements a multi-hop question answering system using dense retrieval, chain-of-evidence construction, and entailment-based verification, inspired by production QA pipelines from Google, Meta, and OpenAI.

Developed an end-to-end QA pipeline inspired by production systems (Google, Meta, OpenAI), implementing dense retrieval with FAISS, chain-of-evidence construction, and cross-encoder entailment verification
## Architecture

The system follows this pipeline:
1. **Question** → Dense Retriever (Bi-Encoder) → Top-k passages
2. **Chain-of-Evidence Builder** → Passage chains (length 1,2,3)
3. **Cross-Encoder Entailment Verifier** → Filtered chains
4. **Answer Extractor/Generator** → Final Answer

## Components

- **Dense Retriever**: Uses sentence-transformers/all-MiniLM-L6-v2 with FAISS for efficient similarity search
- **Chain Builder**: Constructs chains of passages using combinations
- **Entailment Verifier**: Uses roberta-large-mnli for entailment scoring
- **Answer Extraction**: Uses BERT QA head for span extraction

## Dataset

Uses HotpotQA dataset (hotpot_train_v1.1.json, hotpot_dev_distractor_v1.json)

## Ablation Experiments

- Chain length: 1, 2, 3
- Verification threshold: 0.3, 0.5, 0.7, 0.9
- Retriever top-k: 5, 10, 20

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Download HotpotQA data to `data/hotpotqa/`
3. Run pipeline: `python run_pipeline.py`

## Evaluation

See `notebooks/evaluation.ipynb` for metrics and plots.

## CV Description

**Multi-Hop Question Answering System with Entailment Verification**

- Developed an end-to-end QA pipeline inspired by production systems (Google, Meta, OpenAI), implementing dense retrieval with FAISS, chain-of-evidence construction, and cross-encoder entailment verification
- Built using sentence-transformers, transformers, and PyTorch.
- Conducted comprehensive ablation studies on chain length, verification thresholds, and retrieval parameters
- Demonstrated expertise in vector databases (FAISS), NLP models, and multi-hop reasoning
