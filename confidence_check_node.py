class ConfidenceCheckNode:
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def check(self, result):
        return "accepted" if result["confidence"] >= self.threshold else "fallback"
