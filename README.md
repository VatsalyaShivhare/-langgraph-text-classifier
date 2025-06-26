# -langgraph-text-classifier
# 🧠 LangGraph Text Classifier

A robust, self-healing text classification pipeline built using **LangGraph**, **Transformers**, and a fine-tuned **DistilBERT** model. It uses a DAG-based architecture to route inputs through confidence checks and fallback logic, ensuring more reliable predictions.

---

## 🚀 Features

- ✅ Fine-tuned transformer model (DistilBERT)
- ✅ LangGraph-powered DAG pipeline
- ✅ Confidence-based decision routing
- ✅ Fallback mechanism for low-confidence predictions
- ✅ Easy-to-use CLI
- ✅ Logging for inference tracking

---

## 📁 Project Structure

langgraph-text-classifier/
│
├── model/
│ └── fine_tuned_model/ # Your trained model and tokenizer
│
├── src/
│ ├── inference_node.py # Loads the model & performs prediction
│ ├── confidence_check_node.py # Confidence threshold check
│ ├── fallback_node.py # Fallback logic for uncertain cases
│ ├── langgraph_pipeline.py # DAG definition and logic
│ └── cli.py # CLI entry point
│
├── logs/
│ └── run_log.txt # Stores logs of classification runs
│
├── requirements.txt
└── README.md


