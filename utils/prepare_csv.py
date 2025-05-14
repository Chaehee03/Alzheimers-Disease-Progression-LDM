import os
import argparse
import yaml

import pandas as pd
import nibabel as nib
from tqdm import tqdm
from models import const


def make_csv_A(df):
    """
    Creates CSV A, which adds the normalized region volumes in the existing CSV.
    The regional volumes are measured from SynthSeg segmentation and normalized
    using the training samples as a reference.
    """
    coarse_regions = const.COARSE_REGIONS
    code_map = const.SYNTHSEG_CODEMAP

    records = []
    for record in tqdm(df.to_dict(orient='records')):
        segm = nib.load(record['segm_path']).get_fdata().round()
        record['head_size'] = (segm > 0).sum()

        for region in coarse_regions:
            record[region] = 0

        for code, region in code_map.items():
            if region == 'background': continue
            coarse_region = region.replace('left_', '').replace('right_', '')
            record[coarse_region] += (segm == code).sum()

        records.append(record)

    csv_a_df = pd.DataFrame(records)

    for region in coarse_regions:
        # normalize volumes using min-max scaling
        train_values = csv_a_df[csv_a_df.split == 'train'][region]
        minv, maxv = train_values.min(), train_values.max()
        csv_a_df[region] = (csv_a_df[region] - minv) / (maxv - minv)

    return csv_a_df


def make_csv_B(df):
    """
    Creates CSV B, which contains all possible pairs (x_a, x_b) such that
    both scans belong to the same patient and scan x_a is acquired before scan x_b.
    """

    df['acq_date'] = pd.to_datetime(df['acq_date'])

    sorting_field = 'acq_date' if 'acq_date' in df.columns else 'age'

    data = []
    for subject_id in tqdm(df.subject_id.unique()):
        subject_df = df[df.subject_id == subject_id].sort_values(sorting_field, ascending=True)
        for i in range(len(subject_df)):
            for j in range(i + 1, len(subject_df)):
                s_rec = subject_df.iloc[i]
                e_rec = subject_df.iloc[j]
                record = {'subject_id': s_rec.subject_id, 'sex': s_rec.sex, 'split': s_rec.split}
                remaining_columns = set(df.columns).difference(set(record.keys()))
                for column in remaining_columns:
                    record[f'starting_{column}'] = s_rec[column]
                    record[f'followup_{column}'] = e_rec[column]
                data.append(record)
    return pd.DataFrame(data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path',
                        default='config/adni.yaml', type=str)
    args = parser.parse_args()

    # Read the config file #
    with open(args.config_path, 'r') as file:  # open config file
        try:
            config = yaml.safe_load(file)  # YAML file -> python dictionary
        except yaml.YAMLError as exc:  # fail to open config file -> exception
            print(exc)
    print(config)
    ########################

    train_config = config['train_params']

    # read the dataset
    df = pd.read_csv(train_config['dataset_csv'])

    print()
    a_csv_path = os.path.join(train_config['meta_csv'], 'A.csv')
    b_csv_path = os.path.join(train_config['meta_csv'], 'B.csv')


    print('> Creating CSV A\n')
    csv_A = make_csv_A(df)
    csv_A.to_csv(a_csv_path, index=False)

    print('> Creating CSV B\n')
    csv_B = make_csv_B(csv_A)
    csv_B.to_csv(b_csv_path, index=False)

    # if os.path.exists(a_csv_path):
    #     print(f"A.csv already exists at {a_csv_path}")
    # else:
    #     print('> Creating CSV A\n')
    #     csv_A = make_csv_A(df)
    #     csv_A.to_csv(a_csv_path, index=False)
    #
    # if os.path.exists(b_csv_path):
    #     print(f"B.csv already exists at {b_csv_path}")
    # else:
    #     print('> Creating CSV B\n')
    #     csv_A = pd.read_csv(a_csv_path)
    #     csv_B = make_csv_B(csv_A)
    #     csv_B.to_csv(b_csv_path, index=False)