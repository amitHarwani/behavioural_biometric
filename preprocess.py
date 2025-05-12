"""
Preprocessing

Parameters: Time window in seconds, Sequence Length

Loop through users and sequences for a user (Ignoring user 733162 because of 6 sessions of missing accelerometer data)
Store all modality data into dataframes

Call functions to clean the data for each modality: 
    - Keystroke: Remove -ve and >255 keycode rows, Ensure proper sequence (BehaveFormer code), 
    - Accelerometer, Gyr., Mag.: Extract Features (Derivatives, FFT), 
    - Touch: convert event time to numeric, Remove isna rows, Remove rows with incorrect timestamps

Synchronize Data
    - Min. and max. timestamp of key, acc., gyr., mag.
    - Loop from the min. timestamp to max. timestamp in increment of time window:
        - Form combined feature vector for timewindow, from each modality and Extract features, and normalize.

        - Once sequence length is reached combine the feature vectors into one array
    - Output: List of users -> List of sessions -> List of sequences -> sequence length x feat. count

NOTES: 
    - Not Including Pressure In Touch as all values are 1. (Seen in Init. Analysis)

"""

import os
import pandas as pd
import numpy as np
import multiprocessing
import time
import pickle
from sklearn.model_selection import train_test_split



HMOG_DATASET_PATH = './datasets/hmog/public_dataset'

# Parameters
TIME_WINDOW = 10 # Time window in milliseconds


# Combines press and release key events into one row.
def shrink_key_pairs(df: pd.DataFrame):
    length = df.shape[0]

    drop_indices = []
    i = 0
    # Going through the rows
    while(i < length):
        # Getting the release time from the next row
        next_key_event_time = df.iloc[i + 1]["event_time"]
        # Adding it to the current row
        df.at[i, 'release_time'] = next_key_event_time
        # Adding the next rows index to drop_indices list
        drop_indices.append(i + 1)

        i = i + 2

    # Dropping the rows
    df.drop(df.index[drop_indices], inplace=True)
    # Renaming event_time to press_time
    df.rename(columns = {'event_time':'press_time'}, inplace = True)
    # Deleting press_type column and reset index
    del df["press_type"]
    df.reset_index(inplace = True, drop = True)

# Setting: 1 to be pressed and 0 to be released
def check_pair_order(df, index):
    # If the key is released
    if (df.at[index, "press_type"] == 0):
        # If the next keys timestamp is greater than current keys timestamp
        if (df.at[index, "event_time"] < df.at[index + 1, "event_time"]):
            # swap press type
            df.at[index, "press_type"] = 1
            df.at[index + 1, "press_type"] = 0
        # If next keys timestamp is less than current keys timestamp
        else:
            # Swap rows to correct the ordering
            temp = df.loc[index]
            df.loc[index] = df.loc[index + 1]
            df.loc[index + 1] = temp
    # If the key is pressed 
    else:
        # if the keys timestamp is greater than the next keys timestamp
        if (df.at[index, "event_time"] > df.at[index + 1, "event_time"]):
            # swap rows and press types to correct ordering
            df.at[index, "press_type"] = 0
            df.at[index + 1, "press_type"] = 1
            temp = df.loc[index]
            df.loc[index] = df.loc[index + 1]
            df.loc[index + 1] = temp

def clean_keystroke_data(key_csv: pd.DataFrame):

    # Empty Data, return a dataframe with the updated cols list to align with the scenario when dataframe is not empty
    if key_csv.empty: 
        return pd.DataFrame(columns=["press_time", "key_code", "release_time"])
    
     # Dropping data which is not portrait and resetting the indes
    key_csv.drop(index=key_csv[key_csv["orientation"] != 0].index, inplace=True)
    key_csv.reset_index(drop=True, inplace=True)

    # Dropping the orientation column
    key_csv.drop(columns=['orientation'], inplace=True)
    
    # Removing keystrokes where keycode is less than 0 or greater than 255
    # key_csv.drop(index=key_csv[key_csv["key_code"] < 0].index, inplace=True)
    key_csv.drop(index=key_csv[key_csv["key_code"] > 255].index, inplace=True)

    # Resetting the index after dropping
    key_csv.reset_index(drop=True, inplace=True)

    # assert key_csv[key_csv["key_code"] < 0].shape[0] == 0, "Keycodes still Include Negative Values"
    assert key_csv[key_csv["key_code"] > 255].shape[0] == 0, "Keycodes still Include > 255 Values"

    length = key_csv.shape[0]
    index=0

    while(index < length):
        # If its the last row (Odd number of rows)
        if (index == length-1):
            # Add a new row with the opposite press type as the current row
            key_csv.loc[index + 0.5] = [key_csv.at[index, "event_time"], 1 - int(key_csv.at[index, "press_type"]), key_csv.at[index, "key_code"]]
            key_csv = key_csv.sort_index().reset_index(drop=True)
            check_pair_order(key_csv, index)
            break
        else:
            # If the keycodes are different or if the press types are the same
            if ((key_csv.at[index, "key_code"] != key_csv.at[index + 1, "key_code"]) or (key_csv.at[index, "press_type"] == key_csv.at[index + 1, "press_type"])):
                # Add a new row with the opposite press type
                key_csv.loc[index + 0.5] = [key_csv.at[index, "event_time"], 1 - int(key_csv.at[index, "press_type"]), key_csv.at[index, "key_code"]]
                key_csv = key_csv.sort_index().reset_index(drop=True)
                length = key_csv.shape[0]
            check_pair_order(key_csv, index)
            index = index + 2

    assert key_csv.shape[0] % 2 == 0, "Keystroke data doesn't have equal pairs"

    shrink_key_pairs(key_csv)
    return key_csv

def clean_imu_data(imu_csv: pd.DataFrame):
     # Dropping data which is not portrait and resetting the index
    imu_csv.drop(index=imu_csv[imu_csv["orientation"] != 0].index, inplace=True)
    imu_csv.reset_index(drop=True, inplace=True)

    # Dropping the orientation column
    imu_csv.drop(columns=['orientation'], inplace=True)

def clean_touch_data(touch_csv: pd.DataFrame):

     # Dropping data which is not portrait and resetting the index
    touch_csv.drop(index=touch_csv[touch_csv["orientation"] != 0].index, inplace=True)
    touch_csv.reset_index(drop=True, inplace=True)

    # Dropping the orientation column
    touch_csv.drop(columns=['orientation'], inplace=True)

    # Converting event_time column to numeric
    touch_csv["event_time"] = pd.to_numeric(touch_csv["event_time"], errors='coerce')

    # Dropping NA values (Seen in init_analysis that the dataset has 2 rows of NA values across users and sessions)
    touch_csv.dropna(inplace=True, ignore_index=True)

    # Finding event time indices which are not increasing (Seen in init analysis)
    non_increasing_event_times = touch_csv[touch_csv["event_time"].diff() < 0].index.tolist()
    if len(non_increasing_event_times) > 0:
        # Removing incorrect event time rows and resetting the index
        for index in non_increasing_event_times:
            touch_csv.drop(index=(index - 1), inplace=True)
        touch_csv.reset_index(drop=True, inplace=True)

    # Incorrect event time indices, the value provided is the highest valid timestamp in the dataset (Identified in init_analysis)
    indices_with_incorrect_timestamps = touch_csv[touch_csv["event_time"] > 1402767052230.0]["event_time"].sort_index().index

    # If there are indices with incorrect timestamps drop them, and reset the index
    if len(indices_with_incorrect_timestamps) != 0:
        touch_csv.drop(index = indices_with_incorrect_timestamps, inplace=True)
        touch_csv.reset_index(drop=True, inplace=True)

def extract_keystoke_features(key_csv: pd.DataFrame):
    
    # # If the dataframe is empty return all the required cols with 0
    if key_csv.shape[0] == 0:
        cols = ["press_time", "key_code", "release_time","hl", "di_ud", "di_dd", "di_uu", "di_du", "tri_ud", "tri_dd", "tri_uu", "tri_du", "key"]
        return pd.DataFrame([[0.0] * len(cols)], columns=cols)

    # Converting to int to make it generic (Based upon system type)
    key_csv["press_time"] = key_csv["press_time"].astype(int)

    for i in range(key_csv.shape[0]):
        # Hold latency: Release time - press time
        hl = key_csv.at[i, "release_time"] - key_csv.at[i, "press_time"]
        # Key code
        key = key_csv.iloc[i]["key_code"]
        # Digram and tri-gram features
        di_ud = 0.0
        di_dd = 0.0
        di_uu = 0.0
        di_du = 0.0
        tri_ud = 0.0
        tri_dd = 0.0
        tri_uu = 0.0
        tri_du = 0.0

        # Digram
        if (i < key_csv.shape[0] - 1):
            di_ud = key_csv.at[i+1, "press_time"] - key_csv.at[i, "release_time"]
            di_dd = key_csv.at[i+1, "press_time"] - key_csv.at[i, "press_time"]
            di_uu = key_csv.at[i+1, "release_time"] - key_csv.at[i, "release_time"]
            di_du = key_csv.at[i+1, "release_time"] - key_csv.at[i, "press_time"]
        # Trigram
        if (i < key_csv.shape[0] - 2):
            tri_ud = key_csv.at[i+2,"press_time"] - key_csv.at[i, "release_time"]
            tri_dd = key_csv.at[i+2,"press_time"] - key_csv.at[i, "press_time"]
            tri_uu = key_csv.at[i+2,"release_time"] - key_csv.at[i, "release_time"]
            tri_du = key_csv.at[i+2,"release_time"] - key_csv.at[i, "press_time"]
        # Adding features to the dataframe
        key_csv.loc[i, ["hl", "di_ud", "di_dd", "di_uu", "di_du", "tri_ud", "tri_dd", "tri_uu", "tri_du", "key"]] = [hl, di_ud, di_dd, di_uu, di_du, tri_ud, tri_dd, tri_uu, tri_du, key]


    return key_csv

def extract_imu_features(imu_csv: pd.DataFrame, sensor: str):

    edge_order = 2

    # If the dataframe is empty return all the required cols with 0
    if len(imu_csv) == 0:
        cols = ["event_time",f"{sensor}_x", f"{sensor}_y", f"{sensor}_z", f"{sensor}_fd_x", f"{sensor}_fd_y", f"{sensor}_fd_z", 
                f"{sensor}_sd_x", f"{sensor}_sd_y", f"{sensor}_sd_z"]
        return pd.DataFrame([[0.0] * len(cols)], columns=cols )
    
    # Renaming x, y, and z cols to sensor_x, sensor_y, and sensor_z
    imu_csv.rename(columns={"x": f"{sensor}_x", "y": f"{sensor}_y", "z": f"{sensor}_z"}, inplace=True)
    # First and second order derivatives
    imu_csv[f"{sensor}_fd_x"] = np.gradient(imu_csv[f"{sensor}_x"].values, edge_order=edge_order)
    imu_csv[f"{sensor}_fd_y"] = np.gradient(imu_csv[f"{sensor}_y"].values, edge_order=edge_order)
    imu_csv[f"{sensor}_fd_z"] = np.gradient(imu_csv[f"{sensor}_z"].values, edge_order=edge_order)

    imu_csv[f"{sensor}_sd_x"] = np.gradient(imu_csv[f"{sensor}_fd_x"].values, edge_order=edge_order)
    imu_csv[f"{sensor}_sd_y"] = np.gradient(imu_csv[f"{sensor}_fd_y"].values, edge_order=edge_order)
    imu_csv[f"{sensor}_sd_z"] = np.gradient(imu_csv[f"{sensor}_fd_z"].values, edge_order=edge_order)


    return imu_csv

def extract_touch_features(touch_csv: pd.DataFrame):

    edge_order = 2

    # If the dataframe is empty return all the required cols with 0
    if len(touch_csv) == 0:
        cols = ["event_time","t_x", "t_y", "contact_size", "t_fd_x", "t_fd_y", "t_sd_x", "t_sd_y"]
        
        return pd.DataFrame([[0.0] * len(cols)], columns=cols )

    # Renaming x and y cols to t_x and t_y
    touch_csv.rename(columns={"x": "t_x", "y": "t_y"}, inplace=True)
    # First and second order derivatives
    touch_csv["t_fd_x"] = np.gradient(touch_csv["t_x"].values, edge_order=edge_order)
    touch_csv["t_fd_y"] = np.gradient(touch_csv["t_y"].values, edge_order=edge_order)

    touch_csv["t_sd_x"] = np.gradient(touch_csv["t_fd_x"].values, edge_order=edge_order)
    touch_csv["t_sd_y"] = np.gradient(touch_csv["t_fd_y"].values, edge_order=edge_order)

    return touch_csv

# Synchronizing session data by dividing it into multiple sequences
def synchronize_data(key_csv: pd.DataFrame, acc_csv: pd.DataFrame, gyr_csv: pd.DataFrame, mag_csv: pd.DataFrame, touch_csv: pd.DataFrame, TIME_WINDOW: int):

    # Session start and end time = min/max of event times across modalities
    session_start_time = min(df.at[0, "press_time"] if df is key_csv else df.at[0, "event_time"] for df in [key_csv, acc_csv, gyr_csv, mag_csv, touch_csv] if not df.empty)
    session_end_time = max(df.iloc[-1]["press_time"] if df is key_csv else df.iloc[-1]["event_time"] for df in [key_csv, acc_csv, gyr_csv, mag_csv, touch_csv] if not df.empty)

    # Window start time initialized to session start time
    window_start_time = session_start_time

    # Time window in milliseconds
    time_window_in_ms = TIME_WINDOW
    
    sequences = []

    while window_start_time < session_end_time:
        # Window end time = window start + time window length
        window_end_time = window_start_time + time_window_in_ms

        # Data from modalities within the time window
        relevant_key_data = key_csv[(key_csv["press_time"] >= window_start_time) & (key_csv["press_time"] <= window_end_time)].copy().reset_index(drop=True)
        relevant_acc_data = acc_csv[(acc_csv["event_time"] >= window_start_time) & (acc_csv["event_time"] <= window_end_time)].copy().reset_index(drop=True)
        relevant_gyr_data = gyr_csv[(gyr_csv["event_time"] >= window_start_time) & (gyr_csv["event_time"] <= window_end_time)].copy().reset_index(drop=True)
        relevant_mag_data = mag_csv[(mag_csv["event_time"] >= window_start_time) & (mag_csv["event_time"] <= window_end_time)].copy().reset_index(drop=True)
        relevant_touch_data = touch_csv[(touch_csv["event_time"] >= window_start_time) & (touch_csv["event_time"] <= window_end_time)].copy().reset_index(drop=True)

        # If no data exists in this time period, increment window start time and continue
        if relevant_key_data.empty and relevant_acc_data.empty and relevant_gyr_data.empty and relevant_mag_data.empty and relevant_touch_data.empty:
            window_start_time = window_end_time + 1
            continue
        
        # Dropping the press time, release time and key code columns
        relevant_key_data.drop(columns=["press_time", "release_time", "key_code"], inplace=True)
        # Dropping the event time column
        relevant_acc_data.drop(columns=["event_time"], inplace=True)
        relevant_gyr_data.drop(columns=["event_time"], inplace=True)
        relevant_mag_data.drop(columns=["event_time"], inplace=True)
        relevant_touch_data.drop(columns=["event_time"], inplace=True)

        # Empty dataframes are filled with 0s
        if relevant_key_data.empty:
            relevant_key_data.loc[0] = [0.0] * len(relevant_key_data.columns)
        if relevant_acc_data.empty:
            relevant_acc_data.loc[0] = [0.0] * len(relevant_acc_data.columns) 
        if relevant_gyr_data.empty:
            relevant_gyr_data.loc[0] = [0.0] * len(relevant_gyr_data.columns)
        if relevant_mag_data.empty:
            relevant_mag_data.loc[0] = [0.0] * len(relevant_mag_data.columns)
        if relevant_touch_data.empty:
            relevant_touch_data.loc[0] = [0.0] * len(relevant_touch_data.columns)
        
        # If the dataframes have more than one row, take the mean of the rows and convert to a single row dataframe - 
        # Rare because of 10ms time window
        if relevant_key_data.shape[0] > 1:
             relevant_key_data = relevant_key_data.mean(axis = 0).to_frame().T
        if relevant_acc_data.shape[0] > 1:
             relevant_acc_data = relevant_acc_data.mean(axis = 0).to_frame().T
        if relevant_gyr_data.shape[0] > 1:
            relevant_gyr_data = relevant_gyr_data.mean(axis = 0).to_frame().T
        if relevant_mag_data.shape[0] > 1:
            relevant_mag_data = relevant_mag_data.mean(axis = 0).to_frame().T
        if relevant_touch_data.shape[0] > 1:
            relevant_touch_data = relevant_touch_data.mean(axis = 0).to_frame().T

        # Stack this time-window column wise. with one row: (1, 44) -- FFT's will be added later
        time_window_feature = pd.concat([relevant_key_data, relevant_acc_data, relevant_gyr_data, relevant_mag_data, relevant_touch_data], axis=1)

        # Concatenate the windows row wise (Time-Series) and append to sequences list
        sequences.append(time_window_feature.to_numpy())

        # Incrementing start time by 1
        window_start_time = window_end_time + 1

    return sequences


def sync_by_binning(key: pd.DataFrame, acc: pd.DataFrame, gyr: pd.DataFrame, mag: pd.DataFrame, touch: pd.DataFrame, TIME_WINDOW=10):
    # find global start/end (in ms)
    dfs = [key, acc, gyr, mag, touch]
    starts = [
        df["press_time"].iat[0] if (df is key and not df.empty) else
        df["event_time"].iat[0] if not df.empty else np.inf
        for df in dfs
    ]
    ends = [
        df["press_time"].iat[-1] if (df is key and not df.empty) else
        df["event_time"].iat[-1] if not df.empty else -np.inf
        for df in dfs
    ]
    session_start = min(starts)
    session_end   = max(ends)
    
    # helper: assign each row to window = floor((t‑start)/WIN)
    def assign_win(df, col) -> tuple[pd.DataFrame, pd.Series]:
        if df.empty:
            return df, pd.Series(dtype=int)
        w = ((df[col] - session_start) // TIME_WINDOW).astype(int)
        return df, w
    
    k, kw = assign_win(key,   "press_time")
    a, aw = assign_win(acc,   "event_time")
    g, gw = assign_win(gyr,   "event_time")
    m, mw = assign_win(mag,   "event_time")
    t, tw = assign_win(touch, "event_time")

    # drop unused cols
    k = k.drop(columns=["press_time","release_time","key_code"], errors="ignore")
    a = a.drop(columns="event_time")
    g = g.drop(columns="event_time")
    m = m.drop(columns="event_time")
    t = t.drop(columns="event_time")

    # group‑mean by window index
    key_feats   = k.assign(window=kw).groupby("window").mean()
    acc_feats   = a.assign(window=aw).groupby("window").mean()
    gyr_feats   = g.assign(window=gw).groupby("window").mean()
    mag_feats   = m.assign(window=mw).groupby("window").mean()
    touch_feats = t.assign(window=tw).groupby("window").mean()


    # 5) compute the union of all windows where anything happened
    active_windows = (
        key_feats.index
        .union(acc_feats.index)
        .union(gyr_feats.index)
        .union(mag_feats.index)
        .union(touch_feats.index)
        .sort_values()
    )
    key_feats = key_feats.reindex(active_windows, fill_value=0)
    acc_feats = acc_feats.reindex(active_windows, fill_value=0)
    gyr_feats = gyr_feats.reindex(active_windows, fill_value=0)
    mag_feats = mag_feats.reindex(active_windows, fill_value=0)
    touch_feats = touch_feats.reindex(active_windows, fill_value=0)
        

    # concat and return
    all_feats = pd.concat(
        [key_feats, acc_feats, gyr_feats, mag_feats, touch_feats],
        axis=1
    )
    return all_feats


def preprocess_user(user_id):
    user_dir_path = os.path.join(HMOG_DATASET_PATH, user_id)
    sessions = []

    # For each session of the user
    for session in os.listdir(user_dir_path):
        # If its a session
        if 'session' in session:
            print("Starting Session", session)
            # Reading Keystroke Data
            key_csv = pd.read_csv(f"{user_dir_path}/{session}/KeyPressEvent.csv", header=None, usecols=[0, 3, 4, 5], names=["event_time", "press_type", "key_code", "orientation"])
            
            # If the session contains keystroke data
            if not key_csv.empty:
                # Clean Keystroke Data
                key_csv = clean_keystroke_data(key_csv)

                # Reading Sensor Data
                acc_csv = pd.read_csv(f"{user_dir_path}/{session}/Accelerometer.csv", header=None, usecols=[0, 3, 4, 5, 6], names=["event_time", "x", "y", "z", "orientation"])   
                gyr_csv = pd.read_csv(f"{user_dir_path}/{session}/Gyroscope.csv", header=None, usecols=[0, 3, 4, 5, 6], names=["event_time", "x", "y", "z", "orientation"])   
                mag_csv = pd.read_csv(f"{user_dir_path}/{session}/Magnetometer.csv", header=None, usecols=[0, 3, 4, 5, 6], names=["event_time", "x", "y", "z", "orientation"]) 

                clean_imu_data(acc_csv)
                clean_imu_data(gyr_csv)
                clean_imu_data(mag_csv)
                
                # Reading Touch Data
                touch_csv = pd.read_csv(f"{user_dir_path}/{session}/TouchEvent.csv", header=None, usecols=[0, 6, 7, 9, 10], names=["event_time", "x", "y", "contact_size", "orientation"])

                # Clean Touch Data
                clean_touch_data(touch_csv)

                # Extracting Features
                key_csv = extract_keystoke_features(key_csv)
                acc_csv = extract_imu_features(acc_csv, 'a')
                gyr_csv = extract_imu_features(gyr_csv, 'g')
                mag_csv = extract_imu_features(mag_csv, 'm')
                touch_csv = extract_touch_features(touch_csv)

                # Synchronizing data and extracting features
                sequences = sync_by_binning(key_csv, acc_csv, gyr_csv, mag_csv, touch_csv, TIME_WINDOW)

                # Appending to sessions list
                sessions.append(sequences)

                print("Completed Session", session)

    return sessions


def preprocess_data(user_ids):

    users_data = []
    # Preprocess each user sequentially
    # for user_id in user_ids:
    #     users_data.append(preprocess_user(user_id))
    #     break

    # Processing each user in the list parallelly utilizing all the CPU cores available
    with multiprocessing.Pool(CPU_COUNT) as p:
        users_data = p.map(preprocess_user, user_ids)

    return users_data

def main():
    start_time = time.time()

    # List of all the user ids
    user_ids = os.listdir(os.path.join(HMOG_DATASET_PATH))

    # Remove user 733162 based upon init_analysis (Missing Accelerometer Data In 6 sessions)
    user_ids.remove("733162") 

    # Dividing user_ids into train, validation and test 
    user_ids_train, user_ids_val = train_test_split(user_ids, train_size=79, test_size=20, shuffle=True, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)
    user_ids_test, user_ids_val = train_test_split(user_ids_val, train_size=10, test_size=10, shuffle=True, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)

    # Check if validation, test and train users are unique
    assert len(set(user_ids_train) & set(user_ids_val)) == 0, "Train and Val Users are not unique"
    assert len(set(user_ids_train) & set(user_ids_test)) == 0, "Train and Test Users are not unique"
    assert len(set(user_ids_val) & set(user_ids_test)) == 0, "Val and Test Users are not unique"
    assert len(set(user_ids_train) & set(user_ids_val) & set(user_ids_test)) == 0, "Train, Val and Test Users are not unique"

    # Final Output -> List of Users -> List of 8 sessions -> Each having a pandas dataframe of 44 columns, with each row representing 10ms of data.
    # Preprocess Data For Each Split
    training_users_data = preprocess_data(user_ids_train)
    with open(f"v1_training_users_data_tw{TIME_WINDOW}ms.pickle",'wb') as outfile:
        pickle.dump(training_users_data, outfile)
    print("Training Users Done")

    validation_users_data = preprocess_data(user_ids_val)
    with open(f"v1_validation_users_data_tw{TIME_WINDOW}ms.pickle",'wb') as outfile: 
        pickle.dump(validation_users_data, outfile)
    print("Validation Users Done")

    test_users_data = preprocess_data(user_ids_test)
    with open(f"v1_test_users_data_tw{TIME_WINDOW}ms.pickle",'wb') as outfile:
        pickle.dump(test_users_data, outfile)
    print("Test Users Done")

    end_time = time.time()
    print(f"Time Taken: {end_time - start_time}s")




if __name__ == "__main__": 
    TRAIN_TEST_SPLIT_RANDOM_STATE = 25 # Random state for train test split
    CPU_COUNT = multiprocessing.cpu_count() // 2 # Number of CPU Cores available

    main()




