import sys
sys.path.append('..')

from pipeline.qa_pipeline import QAPipeline
import json
from sklearn.metrics import f1_score
import re

def normalize_answer(s):
    """Normalize answer for evaluation"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

def run_ablation_chain_length(dev_data_path, retriever_index_path=None):
    """Ablate chain length"""
    # Load data
    pipeline = QAPipeline()
    dev_data = pipeline.load_data(dev_data_path)
    
    # Limit to first 100 for quick ablation
    dev_data = dev_data[:100]
    
    if retriever_index_path:
        pipeline.retriever.load_index(retriever_index_path)
    else:
        # Prepare passages and encode
        passages = pipeline.prepare_passages(dev_data)
        pipeline.retriever.encode_passages(passages)
    
    results = {}
    for chain_length in [1, 2, 3]:
        print(f"Evaluating chain length: {chain_length}")
        predictions = pipeline.evaluate(dev_data, chain_length=chain_length)
        
        em_scores = [exact_match(p['prediction'], p['gold_answer']) for p in predictions]
        f1_scores = [f1(p['prediction'], p['gold_answer']) for p in predictions]
        
        results[chain_length] = {
            'EM': sum(em_scores) / len(em_scores),
            'F1': sum(f1_scores) / len(f1_scores)
        }
    
    print("Chain Length Ablation Results:")
    for length, scores in results.items():
        print(f"Length {length}: EM={scores['EM']:.3f}, F1={scores['F1']:.3f}")
    
    return results

if __name__ == "__main__":
    run_ablation_chain_length("data/hotpotqa/hotpot_dev_distractor_v1.json")