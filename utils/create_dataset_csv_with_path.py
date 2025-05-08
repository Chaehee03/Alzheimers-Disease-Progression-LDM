import os
import pandas as pd
from tqdm import tqdm

# dataset.csv 불러오기
df = pd.read_csv("dataset.csv")

# image_path 필드가 없으면 추가
if "image_path" not in df.columns:
    df["image_path"] = pd.NA

# segm_path 필드가 없으면 추가
if "segm_path" not in df.columns:
    df["segm_path"] = pd.NA

# image_uid 컬럼을 문자열로 통일
df["image_uid"] = df["image_uid"].astype(str)

root_dir = "/DataRead2/chsong/ADNI1_Complete_3Yr/ADNI_preprocessed"

for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
    for filename in filenames:
        if filename == "normalized.nii.gz":
            # image_uid 추출: 해당 파일의 상위 폴더 이름
            parts = os.path.normpath(dirpath).split(os.sep)
            if len(parts) < 1:
                continue
            image_uid = parts[-1]  # 예: 'I79092'

            if image_uid in df["image_uid"].values:
                full_path = os.path.join(dirpath, filename)
                df.loc[df["image_uid"] == image_uid, "image_path"] = full_path

for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
    for filename in filenames:
        if filename == "segm.nii.gz":
            # image_uid 추출: 해당 파일의 상위 폴더 이름
            parts = os.path.normpath(dirpath).split(os.sep)
            if len(parts) < 1:
                continue
            image_uid = parts[-1]  # 예: 'I79092'

            if image_uid in df["image_uid"].values:
                full_path = os.path.join(dirpath, filename)
                df.loc[df["image_uid"] == image_uid, "segm_path"] = full_path

# 결과 저장
df.to_csv("dataset.csv", index=False)