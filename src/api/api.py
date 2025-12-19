import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware

# ======================================
# ENABLE CORS FOR FRONTEND
# ======================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# LOAD MODELS
# ======================================

TOXIC_MODEL_PATH = "models/xlm_roberta_multilingual"
SENTIMENT_MODEL_PATH = "models/sentiment_model"

tox_tokenizer = AutoTokenizer.from_pretrained(TOXIC_MODEL_PATH)
tox_model = AutoModelForSequenceClassification.from_pretrained(TOXIC_MODEL_PATH)
tox_model.eval()

sent_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
sent_model.eval()

# LABEL MAPS
TOXIC_LABELS = {0: "safe", 1: "offensive", 2: "hate"}
SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}

# ======================================
# INPUT SCHEMA
# ======================================
class InputText(BaseModel):
    text: str

# ======================================
# HELPERS
# ======================================
def run_model(tokenizer, model, text):
    enc = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**enc)
        probs = F.softmax(outputs.logits, dim=1)[0]
        label = torch.argmax(probs).item()
        confidence = probs[label].item()
    return label, confidence


# ======================================
# MAIN API ENDPOINT
# ======================================
@app.post("/analyze")
def analyze_text(data: InputText):

    text = data.text

    # Toxicity model
    t_label, t_conf = run_model(tox_tokenizer, tox_model, text)
    toxic_label = TOXIC_LABELS[t_label]

    # Sentiment model
    s_label, s_conf = run_model(sent_tokenizer, sent_model, text)
    sentiment_label = SENTIMENT_LABELS[s_label]

    # --------------------------------------
    # SPECIAL RULE: Toxic text = negative sentiment
    # --------------------------------------
    if toxic_label != "safe":
        sentiment_label = "negative"

    return {
        "input_text": text,
        "toxicity": {
            "label": toxic_label,
            "confidence": round(float(t_conf), 4)
        },
        "sentiment": {
            "label": sentiment_label,
            "confidence": round(float(s_conf), 4)
        }
    }


@app.get("/")
def home():
    return {"message": "CyberQuash API Running!"}
