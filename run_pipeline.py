#!/usr/bin/env python3

from pipeline.qa_pipeline import QAPipeline
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run Multi-Hop QA Pipeline')
    parser.add_argument('--question', type=str, help='Question to answer')
    parser.add_argument('--data_path', type=str, default='data/hotpotqa/hotpot_dev_distractor_v1.json', help='Path to dev data')
    parser.add_argument('--top_k', type=int, default=10, help='Top-k passages to retrieve')
    parser.add_argument('--chain_length', type=int, default=2, help='Max chain length')
    parser.add_argument('--threshold', type=float, default=0.5, help='Entailment threshold')
    
    args = parser.parse_args()
    
    pipeline = QAPipeline()
    
    # Load and prepare data
    data = pipeline.load_data(args.data_path)
    passages = pipeline.prepare_passages(data)
    pipeline.retriever.encode_passages(passages)
    
    if args.question:
        answer = pipeline.answer_question(args.question, args.top_k, args.chain_length, args.threshold)
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
    else:
        # Evaluate on dev set
        dev_data = test_data  # Small sample
        predictions = pipeline.evaluate(dev_data, args.top_k, args.chain_length, args.threshold)
        for pred in predictions:
            print(f"Q: {pred['question']}")
            print(f"Pred: {pred['prediction']}")
            print(f"Gold: {pred['gold_answer']}")
            print("---")

if __name__ == "__main__":
    main()