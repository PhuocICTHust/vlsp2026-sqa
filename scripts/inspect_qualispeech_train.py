import zipfile
import pandas as pd

ZIP = "data/qualispeech/wav_part1.zip"
CSV = "data/qualispeech/train.csv"

z = zipfile.ZipFile(ZIP)
zip_ids = {
    x.rsplit("/", 1)[-1]
    for x in z.namelist()
    if x.lower().endswith(".wav")
}

df = pd.read_csv(CSV)

mask = df["id"].isin(zip_ids)
train = df[mask].copy()

print("Train CSV:", len(df))
print("Train audio available:", len(train))
print("Missing audio:", (~mask).sum())
print()

print("Overall quality distribution:")
print(train["Overall quality"].value_counts().sort_index())
print()

print("Quality statistics:")
print(train["Overall quality"].describe())
print()

print("Example matched IDs:")
print(train[["id", "Overall quality"]].head(10).to_string(index=False))
