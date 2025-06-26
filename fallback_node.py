class FallbackNode:
    def run(self, result):
        return {
            "label": "uncertain",
            "confidence": result["confidence"],
            "note": "Fallback triggered due to low confidence."
        }
