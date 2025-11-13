import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn
import transformers
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import Trainer, TrainingArguments
import gc

print(np.__version__)
print(transformers.__version__)

# Read the CSV files
df_train = pd.read_csv('training.csv')
df_validation = pd.read_csv('validation.csv')
df_test = pd.read_csv('test.csv')

df_test = df_test.dropna(subset=["text"])

# Display the first 5 rows
print(df_train.head())

zero_count = (df_train['binary-claim'] == 0).sum()
one_count = (df_train['binary-claim'] == 1).sum()

print(f"Number of 0: {zero_count}")
print(f"Number of 1: {one_count}")

"""Imbalanced dataset: texts having no narratives are more than two times of text with narrative."""

print("Percentage of occurrence of 0 and 1:\n")
print(pd.Series(df_train["binary-claim"]).value_counts(normalize=True))

print("First 5 records:\n", df_train.head())

# Ajust dataset for NLI: changing text to premise adding hypothesis
for split in (df_train, df_validation, df_test):
    split.rename(columns={'text': 'premise'}, inplace=True)
    split["hypothesis"] = "This sentence expresses a narrative."
    split.rename(columns={'binary-claim': 'labels'}, inplace=True)
    split["labels"] = 1 - split["labels"] # Flip the labels 0 -> 1, 1 -> 0

# Weighted loss to solve the imbalance problem in data
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=df_train["labels"].to_numpy())  #binary-claim
weights = torch.tensor(class_weights, dtype=torch.float)

# DeBERTa model with class-weighted loss
class WeightedDebertaForSequenceClassification(AutoModelForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = outputs.logits

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {'loss': loss, 'logits': logits}

# Load Tokenizer and Model
#model_name = "microsoft/deberta-v3-large"
model_name = "MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, num_labels=2)

model = WeightedDebertaForSequenceClassification.from_pretrained(model_name, config=config)

# Set class_weights manually
model.class_weights = weights

# Tokenize the data for NLI with hypothesis
train_dataset = Dataset.from_pandas(df_train)
eval_dataset = Dataset.from_pandas(df_validation)
test_dataset = Dataset.from_pandas(df_test)

def tokenize_function(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=256        # or 512
    )

tokenized_train = Dataset.from_pandas(df_train).map(tokenize_function, batched=True)
tokenized_eval  = Dataset.from_pandas(df_validation).map(tokenize_function, batched=True)
tokenized_test  = Dataset.from_pandas(df_test).map(tokenize_function, batched=True)

# drop the original text columns
cols_to_remove = ["premise", "hypothesis", "claim"]     # adjust if "claim" absent
tokenized_train = tokenized_train.remove_columns([c for c in cols_to_remove if c in tokenized_train.column_names])
tokenized_eval  = tokenized_eval.remove_columns([c for c in cols_to_remove if c in tokenized_eval.column_names])
tokenized_test  = tokenized_test.remove_columns([c for c in cols_to_remove if c in tokenized_test.column_names])

tokenized_train.set_format("torch")
tokenized_eval.set_format("torch")
tokenized_test.set_format("torch")

#Trainer and TrainingArguments
training_args = TrainingArguments(
    output_dir="./narrative_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
)

trainer.train()

train_preds = trainer.predict(tokenized_train)

torch.cuda.empty_cache()
gc.collect()

test_preds = trainer.predict(tokenized_test)

# Convert logits to predicted class (0 or 1)
test_logits = test_preds.predictions
test_pred_labels = np.argmax(test_logits, axis=1)

# Ground-truth labels
true_labels = test_preds.label_ids

# Result of adding hypothesis and finetuning "MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary"
print(classification_report(true_labels, test_pred_labels, digits=4))
print("Accuracy:", accuracy_score(true_labels, test_pred_labels))