
import os
import pandas as pd
import shutil
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# HAM10000 metadata expected at data/raw/HAM10000_metadata.csv
meta_path = RAW_DIR / "HAM10000_metadata.csv"
if not meta_path.exists():
    raise FileNotFoundError(
        f"Metadata not found at {meta_path}. Place HAM10000 images and metadata in data/raw/."
    )

df = pd.read_csv(meta_path)

# Classes
classes = sorted(df["dx"].unique().tolist())  # 7 classes

# Build absolute image paths (two source folders in HAM10000 release)
def find_image_path(image_id: str):
    # Images live in two parts: HAM10000_images_part_1 and _part_2
    for p in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        candidate = RAW_DIR / p / f"{image_id}.jpg"
        if candidate.exists():
            return candidate
    return None

df["img_path"] = df["image_id"].apply(find_image_path)
missing = df["img_path"].isna().sum()
if missing > 0:
    raise FileNotFoundError(f"{missing} images not found. Check raw folders.")

# We will do grouped stratification by patient_id if available to avoid leakage
group_col = "lesion_id" if "lesion_id" in df.columns else None
y = df["dx"].values
groups = df[group_col].values if group_col else df["image_id"].values

# Split into train/val/test using StratifiedGroupKFold: 3 splits, pick one for test, one for val.
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(sgkf.split(df, y=y, groups=groups))

# Use first split for train/test, then split train into train/val on another fold
train_idx, test_idx = splits[0]
df_train_all = df.iloc[train_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

# Now create val from train_all
y_tr = df_train_all["dx"].values
groups_tr = df_train_all[group_col].values if group_col else df_train_all["image_id"].values
sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=123)
tr_idx, val_idx = next(sgkf2.split(df_train_all, y=y_tr, groups=groups_tr))

df_train = df_train_all.iloc[tr_idx].reset_index(drop=True)
df_val = df_train_all.iloc[val_idx].reset_index(drop=True)

def copy_split(split_df, split_name):
    for cls in classes:
        (PROC_DIR / split_name / cls).mkdir(parents=True, exist_ok=True)
    for _, row in split_df.iterrows():
        src = row["img_path"]
        cls = row["dx"]
        dst = PROC_DIR / split_name / cls / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

print("Copying train...")
copy_split(df_train, "train")
print("Copying val...")
copy_split(df_val, "val")
print("Copying test...")
copy_split(df_test, "test")
print("Done. Data prepared at data/processed/.")
