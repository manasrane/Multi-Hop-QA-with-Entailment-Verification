from retriever.bi_encoder import BiEncoderRetriever
from verifier.entailment_model import EntailmentVerifier
from pipeline.build_chains import build_chains
from transformers import pipeline
from datasets import load_dataset
import json

class QAPipeline:
    def __init__(self, retriever_model='sentence-transformers/all-MiniLM-L6-v2', 
                 verifier_model='roberta-large-mnli', qa_model='bert-base-uncased'):
        self.retriever = BiEncoderRetriever(retriever_model)
        self.verifier = EntailmentVerifier(verifier_model)
        self.qa_pipeline = pipeline("question-answering", model=qa_model)

    def load_data(self, data_path):
        """Load HotpotQA data"""
        dataset = load_dataset("hotpot_qa", "distractor")
        if "train" in data_path:
            return list(dataset["train"])
        else:
            return list(dataset["validation"])

    def prepare_passages(self, data):
        """Extract all passages from dataset"""
        passages = []
        for item in data:
            context = item['context']
            if isinstance(context, str):
                context = json.loads(context)
            if isinstance(context, dict):
                for para_list in context['sentences']:
                    passages.extend(para_list)
            elif isinstance(context, list):
                for title, paras in context:
                    passages.extend(paras)
            else:
                print(f"Unknown context type: {type(context)}")
        return list(set(passages))  # Remove duplicates

    def answer_question(self, question, top_k=10, chain_length=2, threshold=0.5):
        """Full QA pipeline"""
        # Retrieve passages
        retrieved_passages, _ = self.retriever.retrieve(question, top_k=top_k)
        
        # Build chains
        chains = build_chains(retrieved_passages, max_chain_length=chain_length)
        
        # Verify chains
        verified_chains = self.verifier.filter_chains(chains, question, threshold=threshold)
        
        if not verified_chains:
            return "No sufficient evidence found."
        
        # Get best chain
        best_chain, _ = max(verified_chains, key=lambda x: x[1])
        chain_text = " ".join(best_chain)
        
        # Extract answer
        answer = self.qa_pipeline(question=question, context=chain_text)
        return answer['answer']

    def evaluate(self, dev_data, top_k=10, chain_length=2, threshold=0.5):
        """Evaluate on dev set"""
        predictions = []
        for item in dev_data:
            pred_answer = self.answer_question(item['question'], top_k, chain_length, threshold)
            predictions.append({
                'question': item['question'],
                'prediction': pred_answer,
                'gold_answer': item['answer']
            })
        return predictions