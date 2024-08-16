# Method below is based on make_opportunity

"""
Preprocess McRoberts data into 30s and 10s windows
with a 15s and 5 sec overlap respectively
Sample Rate: 50 Hz

For a typical challenge, 10 sec sliding window and 50%
overlap is used. Either specific sets or runs are used
for training or sometimes both run count and subject count are specified.

Usage:
    python data_processing.py

"""

import numpy as np
import os
import pandas as pd

from statsmodels.nonparametric.smoothers_lowess import lowess
from joblib import Parallel, delayed
from tqdm import tqdm

label2code = {
    'no activity':0,
    'sitting':1,
    'standing':2,
    'walking':3,
    'running':4,
    'cycling':5,
    'stair_down':6,
    'stair_up':7,
    'elevator_down':8,
    'elevator_up':9,
    'escalator_down':10,
    'escalator_up':11,
    'dragging':12,
    'kicking':13,
    'punching':14,
    'throwing':15,
    'bus':16,
    'car':17,
    'tram':18,
    'train':19
}
code2label = {code: label for label, code in label2code.items()}


def content2x_and_y(data_content, epoch_len=30, sample_rate=100, overlap=15):
    sample_count = int(np.floor(len(data_content) / (epoch_len * sample_rate)))

    sample_x_back_acc_idx = 0
    sample_y_back_acc_idx = 1
    sample_z_back_acc_idx = 2
    sample_x_back_gyr_idx = 3
    sample_y_back_gyr_idx = 4
    sample_z_back_gyr_idx = 5
    sample_x_arm_acc_idx = 7
    sample_y_arm_acc_idx = 8
    sample_z_arm_acc_idx = 9
    sample_x_arm_gyr_idx = 10
    sample_y_arm_gyr_idx = 11
    sample_z_arm_gyr_idx = 12
    sample_back_atm_idx = 6


    sample_limit = sample_count * epoch_len * sample_rate
    data_content = data_content[:sample_limit, :]

    x_back_acc = data_content[:, sample_x_back_acc_idx]
    y_back_acc = data_content[:, sample_y_back_acc_idx]
    z_back_acc = data_content[:, sample_z_back_acc_idx]
    x_back_gyr = data_content[:, sample_x_back_gyr_idx]
    y_back_gyr = data_content[:, sample_y_back_gyr_idx]
    z_back_gyr = data_content[:, sample_z_back_gyr_idx]
    x_arm_acc = data_content[:, sample_x_arm_acc_idx]
    y_arm_acc = data_content[:, sample_y_arm_acc_idx]
    z_arm_acc = data_content[:, sample_z_arm_acc_idx]
    x_arm_gyr = data_content[:, sample_x_arm_gyr_idx]
    y_arm_gyr = data_content[:, sample_y_arm_gyr_idx]
    z_arm_gyr = data_content[:, sample_z_arm_gyr_idx]
    back_atm = data_content[:, sample_back_atm_idx]


    # to make overlappting window
    offset = overlap * sample_rate
    shifted_x_back_acc = data_content[offset:-offset:, sample_x_back_acc_idx]
    shifted_y_back_acc = data_content[offset:-offset:, sample_y_back_acc_idx]
    shifted_z_back_acc = data_content[offset:-offset:, sample_z_back_acc_idx]
    shifted_x_back_gyr = data_content[offset:-offset:, sample_x_back_gyr_idx]
    shifted_y_back_gyr = data_content[offset:-offset:, sample_y_back_gyr_idx]
    shifted_z_back_gyr = data_content[offset:-offset:, sample_z_back_gyr_idx]
    shifted_x_arm_acc = data_content[offset:-offset:, sample_x_arm_acc_idx]
    shifted_y_arm_acc = data_content[offset:-offset:, sample_y_arm_acc_idx]
    shifted_z_arm_acc = data_content[offset:-offset:, sample_z_arm_acc_idx]
    shifted_x_arm_gyr = data_content[offset:-offset:, sample_x_arm_gyr_idx]
    shifted_y_arm_gyr = data_content[offset:-offset:, sample_y_arm_gyr_idx]
    shifted_z_arm_gyr = data_content[offset:-offset:, sample_z_arm_gyr_idx]
    shifted_back_atm = data_content[offset:-offset:, sample_back_atm_idx]


    shifted_x_back_acc = shifted_x_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_back_acc = shifted_y_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_back_acc = shifted_z_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_back_gyr = shifted_x_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_back_gyr = shifted_y_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_back_gyr = shifted_z_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_arm_acc = shifted_x_arm_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_arm_acc = shifted_y_arm_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_arm_acc = shifted_z_arm_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_arm_gyr = shifted_x_arm_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_arm_gyr = shifted_y_arm_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_arm_gyr = shifted_z_arm_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_back_atm = shifted_back_atm.reshape(-1, epoch_len * sample_rate, 1)
    

    shifted_X = np.concatenate([shifted_x_back_acc, shifted_y_back_acc, shifted_z_back_acc,
                                shifted_x_back_gyr, shifted_y_back_gyr, shifted_z_back_gyr,
                                shifted_x_arm_acc,  shifted_y_arm_acc,  shifted_z_arm_acc,
                                shifted_x_arm_gyr,  shifted_y_arm_gyr,  shifted_z_arm_gyr,
                                shifted_back_atm], axis=2)

    x_back_acc = x_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    y_back_acc = y_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    z_back_acc = z_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    x_back_gyr = x_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    y_back_gyr = y_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    z_back_gyr = z_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    x_arm_acc = x_arm_acc.reshape(-1, epoch_len * sample_rate, 1)
    y_arm_acc = y_arm_acc.reshape(-1, epoch_len * sample_rate, 1)
    z_arm_acc = z_arm_acc.reshape(-1, epoch_len * sample_rate, 1)
    x_arm_gyr = x_arm_gyr.reshape(-1, epoch_len * sample_rate, 1)
    y_arm_gyr = y_arm_gyr.reshape(-1, epoch_len * sample_rate, 1)
    z_arm_gyr = z_arm_gyr.reshape(-1, epoch_len * sample_rate, 1)
    back_atm = back_atm.reshape(-1, epoch_len * sample_rate, 1)


    X = np.concatenate([x_back_acc, y_back_acc, z_back_acc,
                        x_back_gyr, y_back_gyr, z_back_gyr,
                        x_arm_acc,  y_arm_acc,  z_arm_acc,
                        x_arm_gyr,  y_arm_gyr,  z_arm_gyr,
                        back_atm], axis=2)

    X = np.concatenate([X, shifted_X])
    return X


def process_row(row):
    # Apply LOWESS smoothing
    smoothed_row = lowess(row, np.arange(len(row)), frac=0.5)[:, 1]
    return smoothed_row


def post_process(X):
    # set mean pressure of each window to 0
    means = X[:, :, -1].mean(axis=1)
    X[:, :, -1] -= means[:, None]

    # Parallel processing for LOWESS
    print("Smoothing pressure values...")
    X[:, :, -1] = np.array(Parallel(n_jobs=-1)(delayed(process_row)(X[i, :, -1]) for i in tqdm(range(X.shape[0]))))

    return X


def process_all(file_path, X_path, epoch_len, overlap):
    sample_rate = 50

    # read in data
    datContent = pd.read_csv(file_path, 
                                sep=",",
                                parse_dates=["time"],
                                index_col="time").iloc[:,1:]

    # datContent["label activity"] = datContent["label activity"].map(label2code)
    datContent = datContent.to_numpy()

    current_X = content2x_and_y(
        datContent,
        sample_rate=sample_rate,
        epoch_len=epoch_len,
        overlap=overlap,
    )
    
    if current_X.shape[0] == 0:
        print("Data insufficient for processing")
        return 0
    X = current_X

    # post-process
    X = post_process(X)

    np.save(X_path, X)

    # print some dataset stats
    print("X shape:", X.shape)
    

def get_write_paths(data_root):
    X_path = os.path.join(data_root, "X.npy")

    # Make folder to store X and y if it does not already exist
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    return X_path


# In main file, will iterate thorugh list of desired files, so this will process one file per go
# File1->process->inference->File2->process->inference->etc.
def process_file(
        experiment_id: str
        ):
    cwd = os.getcwd()
    data_root = os.path.join(cwd,"mcroberts_files")
    # Where to find raw data
    files_to_process = [
        filename for filename in os.listdir(data_root)
        if filename.startswith(experiment_id)
    ][0]
    files_to_process = os.path.join(data_root,files_to_process)
    
    # Where to save processed data
    print(f"Processing data for experiment {experiment_id}")
    processed_path = os.path.join(cwd,"processed_data",experiment_id)
    X_path = get_write_paths(processed_path)
    
    window_len = 10
    overlap = 5
    process_all(files_to_process, X_path, window_len, overlap)
    print("Saved data to:", X_path)


if __name__ == "__main__":
    process_file(experiment_id="20230628")
