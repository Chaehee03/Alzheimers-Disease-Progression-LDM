import pandas as pd
import numpy as np

# ---------- 1. 원본 CSV 읽기 ----------
df = pd.read_csv("ADNI1_Complete_3Yr_1.5T_4_26_2025.csv")

# ---------- 2. 열 이름 통일 & 필요한 열만 추리기 ----------
df = df.rename(columns={
    "Subject": "subject_id",
    "Image Data ID": "image_uid",
    "Sex": "sex",
    "Age": "age",
    "Acq Date": "acq_date",
    "Group": "diagnosis"
})[["subject_id", "image_uid", "sex", "age", "acq_date", "diagnosis"]]

# ---------- 3. sex, diagnosis 매핑 ----------
sex_map  = {"Male": 0, "M": 0, "Female": 1, "F": 1}
diag_map = {"CN": 0, "Normal": 0, "MCI": 0.5, "AD": 1}

df["sex"]       = df["sex"].str.strip().map(sex_map)
df["diagnosis"] = df["diagnosis"].str.strip().map(diag_map)

# ---------- 4. age 정규화 ----------
df["age"] = df["age"] / 100.0

# ---------- 5. 날짜 형식 변환 ----------
df["acq_date"] = pd.to_datetime(df["acq_date"], format="%m/%d/%Y")

# ---------- 6. split 할당 (subject 단위 70:15:15) ----------
rng = np.random.default_rng(seed=42)          # 재현 가능성 확보
subjects = df["subject_id"].unique()
rng.shuffle(subjects)

n_sub = len(subjects)
train_end = int(n_sub * 0.70)
valid_end = train_end + int(n_sub * 0.15)

split_map = {**{s: "train" for s in subjects[:train_end]},
             **{s: "valid" for s in subjects[train_end:valid_end]},
             **{s: "test"  for s in subjects[valid_end:]}}

df["split"] = df["subject_id"].map(split_map)

# ---------- 7. last_diagnosis 계산 ----------
last_diag = (df.sort_values(["subject_id", "acq_date"])
               .groupby("subject_id")["diagnosis"]
               .last())
df["last_diagnosis"] = df["subject_id"].map(last_diag)

# ---------- 8. latent_path 초기화 ----------
df["latent_path"] = np.nan

# ---------- 9. 열 순서 정리 & 저장 ----------
df = df[["subject_id", "image_uid", "split", "sex", "age",
         "acq_date", "diagnosis", "last_diagnosis", "latent_path"]]

df.to_csv("dataset.csv", index=False)
print(f"dataset.csv saved – {len(df)} rows")
