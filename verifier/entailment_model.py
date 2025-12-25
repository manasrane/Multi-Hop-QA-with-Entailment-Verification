from transformers import pipeline, AutoTokenizer
import torch

class EntailmentVerifier:
    def __init__(self, model_name='roberta-large-mnli'):
        self.device = 0 if torch.cuda.is_available() else -1
        self.nli_pipeline = pipeline("text-classification", model=model_name, device=self.device, return_all_scores=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 512  # RoBERTa max

    def verify_chain(self, chain_text, question, candidate_answer=None):
        """Verify if chain entails the answer to the question"""
        if candidate_answer:
            hypothesis = f"The answer to the question '{question}' is: {candidate_answer}"
        else:
            hypothesis = f"This information contains enough information to answer the question: {question}"
        
        premise = " ".join(chain_text) if isinstance(chain_text, list) else chain_text
        
        # Truncate premise to fit model
        tokens = self.tokenizer.encode(premise, add_special_tokens=False)
        if len(tokens) > self.max_length - len(self.tokenizer.encode(hypothesis, add_special_tokens=False)) - 3:  # for special tokens
            tokens = tokens[:self.max_length - len(self.tokenizer.encode(hypothesis, add_special_tokens=False)) - 3]
        premise = self.tokenizer.decode(tokens)
        
        result = self.nli_pipeline({"text": premise, "text_pair": hypothesis})
        
        # Get entailment score
        entailment_score = next((item['score'] for item in result if item['label'] == 'ENTAILMENT'), 0.0)
        return entailment_score

    def filter_chains(self, chains, question, threshold=0.5):
        """Filter chains based on entailment threshold"""
        filtered_chains = []
        for chain in chains:
            score = self.verify_chain(chain, question)
            if score >= threshold:
                filtered_chains.append((chain, score))
        return filtered_chains