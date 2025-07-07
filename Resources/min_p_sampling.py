from transformers import LogitsProcessor
import torch

# Custom Logits Processor for Min-P Sampling
class MinPLogitsProcessor(LogitsProcessor):
    def __init__(self, min_p: float):
        self.min_p = min_p

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)
        # Get the maximum probability for each token
        max_prob, _ = torch.max(probs, dim=-1, keepdim=True) 
        # Calculate the threshold based on min_p
        threshold = self.min_p * max_prob
        # Create a mask for probabilities that meet or exceed the threshold 
        mask = probs >= threshold
        
        # Filter the scores: keep original logits for valid tokens, set others to -inf
        filtered_scores = torch.where(mask, scores, torch.tensor(float('-inf'), device=scores.device))
        # Check if all tokens are filtered out (all -inf)
        if torch.all(torch.isinf(filtered_scores)):
            #print("Warning: All tokens filtered out by min-p, returning original scores", file=sys.stderr, flush=True)
            return scores
        # Apply softmax to get normalized probabilities
        normalized_probs = torch.softmax(filtered_scores, dim=-1)
        # Convert back to logits
        scores = torch.log(normalized_probs + 1e-10)  # Add small constant to avoid log(0)
        # Check for invalid values (nan or inf)
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            #print("Warning: Invalid logits detected after softmax, returning original scores", file=sys.stderr, flush=True)
            return scores
        return scores
