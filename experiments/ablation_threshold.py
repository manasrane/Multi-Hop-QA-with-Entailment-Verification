import sys
sys.path.append('..')

from pipeline.qa_pipeline import QAPipeline
import json
from experiments.ablation_chain_length import exact_match, f1

def run_ablation_threshold(dev_data_path, retriever_index_path=None):
    """Ablate verification threshold"""
    # Load data
    pipeline = QAPipeline()
    dev_data = pipeline.load_data(dev_data_path)
    
    # Limit to first 100
    dev_data = dev_data[:100]
    
    if retriever_index_path:
        pipeline.retriever.load_index(retriever_index_path)
    else:
        passages = pipeline.prepare_passages(dev_data)
        pipeline.retriever.encode_passages(passages)
    
    results = {}
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        print(f"Evaluating threshold: {threshold}")
        predictions = pipeline.evaluate(dev_data, threshold=threshold)
        
        em_scores = [exact_match(p['prediction'], p['gold_answer']) for p in predictions]
        f1_scores = [f1(p['prediction'], p['gold_answer']) for p in predictions]
        
        # Count chains kept (approximate)
        chains_kept = len([p for p in predictions if p['prediction'] != "No sufficient evidence found."])
        
        results[threshold] = {
            'EM': sum(em_scores) / len(em_scores),
            'F1': sum(f1_scores) / len(f1_scores),
            'Chains Kept': chains_kept / len(predictions)
        }
    
    print("Threshold Ablation Results:")
    for thresh, scores in results.items():
        print(f"Threshold {thresh}: EM={scores['EM']:.3f}, F1={scores['F1']:.3f}, Chains Kept={scores['Chains Kept']:.3f}")
    
    return results

if __name__ == "__main__":
    run_ablation_threshold("data/hotpotqa/hotpot_dev_distractor_v1.json")