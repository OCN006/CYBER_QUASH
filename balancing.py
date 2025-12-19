import pandas as pd

INPUT = "data/processed/sentiment_multilingual.csv"
OUTPUT = "data/processed/sentiment_balanced.csv"

# How many samples per language per class?
SAMPLES_PER_CLASS = 40000  

print("📥 Loading dataset...")
df = pd.read_csv(INPUT)

print("🔍 Original size:", len(df))

balanced_frames = []

languages = df["lang"].unique()
classes = df["label"].unique()

for lang in languages:
    for cls in classes:
        subset = df[(df["lang"] == lang) & (df["label"] == cls)]
        
        # If fewer samples exist, keep all
        take = min(SAMPLES_PER_CLASS, len(subset))
        
        sampled = subset.sample(take, random_state=42)
        balanced_frames.append(sampled)

balanced_df = pd.concat(balanced_frames, ignore_index=True)

print("📊 New size:", len(balanced_df))

balanced_df = balanced_df.sample(frac=1, random_state=42)  # shuffle
balanced_df.to_csv(OUTPUT, index=False)

print("✅ Saved balanced dataset →", OUTPUT)
