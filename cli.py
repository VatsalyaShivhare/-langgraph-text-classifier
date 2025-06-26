from langgraph_pipeline import ClassificationDAG

def main():
    dag = ClassificationDAG()
    print("🧠 Self-Healing Classifier\nType 'exit' to quit.\n")
    while True:
        text = input("Input: ")
        if text.lower() == "exit":
            break
        # ✅ wrap input in a dictionary
        result = dag.run({"text": text})
        print(f"> Label: {result['label']}\n> Confidence: {round(result['confidence'], 2)}\n")

if __name__ == "__main__":
    main()
