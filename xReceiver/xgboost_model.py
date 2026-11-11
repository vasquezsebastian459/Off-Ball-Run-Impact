import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
from ori_config import OUTPUTS_DIR  # noqa: E402
import joblib
import pandas as pd
import json 
import os
import pickle

import sys
import os

# Ensure the root of the project is in the Python path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

# from xReceiver.xR_get_features_v2 import xR_Run_Processor
from .xR_get_features_v2 import xR_Run_Processor

from .visual import Visualizer
class XGBoostModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.features_cols = self.model.get_booster().feature_names

        self.xgb_run_processor = xR_Run_Processor(self)

        self.visualizer = Visualizer(self)



    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)
        
    def _load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load_data(self, match_id):

        match_dir = f"{OUTPUTS_DIR}/{match_id}"

        # Get merged tracking data

        merged_path = os.path.join(match_dir, "merged_tracking_events.jsonl")

        self.merged_jsonl = []
        with open(merged_path, 'r') as f:
            for line in f:
                self.merged_jsonl.append(json.loads(line))
    

        # Get WyId2TeamId
        self.wy_id2team_id = self._load_pickle(os.path.join(match_dir, "wy_id2team_id.pkl"))

        # Get WyId2PlyrId
        self.wy_id2plyr_id = self._load_pickle(os.path.join(match_dir, "wy_id2plyr_id.pkl"))

        # Get Playr Look up Dict
        self.plyr_lookup_dict = self._load_json(os.path.join(match_dir, "plyr_lookup_dict.json"))

        # Get Match Info
        self.match_info = self._load_json(os.path.join(match_dir, "cached_data/match_info.json"))


    
    def get_predictions_frame(self, row_num):

        predictions = {}

        row = self.merged_jsonl[row_num]

        wy_team_id = row['team']['id']

        # Turn team into DB id

        dl_team_id = self.wy_id2team_id[wy_team_id]

        players_dict = row['data'][str(dl_team_id)]

        # Get the giver and receiver ids
        giver_id = row['player']['id']
        receiver_id = row['pass']['recipient']['id']

        giver_id = self.wy_id2plyr_id[giver_id]
        receiver_id = self.wy_id2plyr_id[receiver_id]

        for plyr in players_dict:
            plyr_id = plyr['id']

            features = self.xgb_run_processor.get_features(plyr_id=plyr_id, type='start')

            if int(plyr_id) == int(giver_id):
                continue

            if int(plyr_id) == int(receiver_id):
                receiver = True
            else:
                receiver = False
    
            
            xP_pred, class_pred = self.predict(features)

            predictions[plyr_id] = {
                'xP_pred': xP_pred,
                'class_pred': class_pred,
                'receiver' : receiver
            }

            

        return predictions

            
    

    def predict(self, features):

        X = features.copy()


        X_df = pd.DataFrame([X])

        X_df = X_df[self.features_cols]


        # Predict the probability of pass success (xP)
        xP_pred = self.model.predict_proba(X_df)[:, 1][0]

        # Predict the class (success/failure)
        class_pred = self.model.predict(X_df)[0]

        return xP_pred, class_pred


if __name__ == "__main__":
    model = XGBoostModel(model_path="stats/xReceiver/models/xgb_model_no_pc.pkl")
    model.load_data(match_id=666385)



    predictions = model.get_predictions_frame(row_num=45096)

    # print(predictions)
    # features = model.get_features(row_num=20, plyr_id=1159683)
    # xR_pred, class_pred = model.predict(features)
    # print(f"Expected Receiver prob {xR_pred}")
    # print(f"Class prediction {class_pred}")