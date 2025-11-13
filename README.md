# Binary Narrative Prediction with DeBERTa
Classifying whether a given sentence expresses a narrative or not.

This repository contains code for **binary narrative prediction** using a **DeBERTa-based sequence classification model** with **class-weighted loss** to handle imbalanced datasets. The model has been applied on the **CARDS** and **Climate_Fever** datasets. The predicted label of 0 means the sentence is narrative, and if the predicted label is 1, the sentence is not narrative. The approach uses:

- **DeBERTa model** (`MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary`) for sequence classification.
- **Weighted cross-entropy loss** to account for class imbalance.
- Conversion of the task into a **Natural Language Inference (NLI)** format by pairing each text with a fixed hypothesis: `"This sentence expresses a narrative."`

---

### Evaluation Metrics

accuracy, precision, recall, f1-score, macro-avg, weighted-avg

---

## Requirements

```bash
pip install transformers datasets scikit-learn
pip install --upgrade transformers
pip install numpy==1.26.4
