from fastapi import FastAPI
app = FastAPI(title="Testing Fast API", version="1.0.0")

@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

MODEL_PATH = "C:\\Users\\Suresh\\Desktop\\DSML\\Projects\\DS-P1\\Day-58 VSCode\\model"

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
@app.post("/classify_review/")

def classify_review(
        user_input: str,
) -> str:
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Tokenize the input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits).item()

    # Map the predicted class ID to a label (assuming binary classification)
    label_map = {0: "Negative", 1: "Positive"}
    predicted_label = label_map.get(predicted_class_id, "Unknown")

    return predicted_label


