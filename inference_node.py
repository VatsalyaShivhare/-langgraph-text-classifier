# src/inference_node.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from pathlib import Path

class InferenceNode:
    def __init__(self):
        # This path construction is excellent and robust. No changes needed here.
        model_path = Path(__file__).resolve().parent.parent / "model" / "fine_tuned_model"
        
        # Load model and tokenizer from the local directory.
        # It's better to remove `local_files_only=True` to make loading more flexible.
        # The library will prioritize local files anyway.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set the model to evaluation mode. This is a best practice.
        self.model.eval()

    def run(self, state: dict):
        """
        Runs the inference model on the text from the state.
        
        Args:
            state (dict): The current state of the graph. 
                          Expected to contain a 'text' key.
                          
        Returns:
            dict: A dictionary containing the 'label' and 'confidence' to be
                  merged back into the state.
        """
        print("---INFERENCE---")
        text = state["text"]
        
        # Tokenize and run inference
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Process the output
        probs = F.softmax(outputs.logits, dim=1)
        confidence, label_id = torch.max(probs, dim=1)
        label = self.model.config.id2label.get(label_id.item(), "UNKNOWN")
        
        # Return a dictionary to update the state
        return {
            "label": label,
            "confidence": confidence.item()
        }