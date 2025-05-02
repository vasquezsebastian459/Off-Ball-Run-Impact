import numpy as np
import pickle

# data = np.load('stats/pitch_control/dfs/PPCF_full_game.npz', allow_pickle=True)
# data = np.load('stats/pitch_control/dfs/622212_PPCF_full_game.npz', allow_pickle=True)
data = np.load('stats/pitch_control/dfs/622207_PPCF_full_game.npz', allow_pickle=True)

# data = np.load('/Users/svasquez/LaurieOnTracking/pitch_control/dfs/test_PPCF_full_game.npz', allow_pickle=True)
# data = np.load('/Users/svasquez/driblab/dl-vision/outputs/666385/PPCF_filtered_30x20.npz', allow_pickle=True)


npz_path = f"/Users/svasquez/driblab/dl-vision/outputs/622126/PPCF_filtered_30x20.npz"
npz_path = f"/Users/svasquez/driblab/dl-vision/outputs/622210/PPCF_filtered_30x20.npz"
data = np.load(npz_path, allow_pickle=True)

# TEST PICKLE

pickle_path = f"outputs/622126/wy_id2plyr_id.pkl"

with open(pickle_path, 'rb') as f:
    pkl_data = pickle.load(f)





print(data.files)

PPCF_data = data['PPCF_data']


# print(PPCF_data)

# Unwrap the actual list
PPCF_data_list = PPCF_data.tolist()


count = len(PPCF_data_list)

for frame in PPCF_data_list:
    print(frame.get('frame_index'))

print(count)

total = 90* 60

print(total)

print(count/total)

# for frame in PPCF_data_list:
#     print(frame)

