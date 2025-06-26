# Example DAG runner
from inference_node import InferenceNode
from confidence_check_node import ConfidenceCheckNode
from fallback_node import FallbackNode
import logging
from datetime import datetime
import os
import logging

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(filename='logs/run_log.txt', level=logging.INFO)

class ClassificationDAG:
    def __init__(self):
        self.infer = InferenceNode()
        self.check = ConfidenceCheckNode()
        self.fallback = FallbackNode()

    def run(self, text):
        result = self.infer.run(text)
        status = self.check.check(result)
        final = result if status == "accepted" else self.fallback.run(result)

        logging.info(f"{datetime.now()} | Input: {text} | Result: {final}")
        return final
