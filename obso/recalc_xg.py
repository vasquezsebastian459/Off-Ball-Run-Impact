import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
from ori_config import S3_FETCH_SCRIPT  # noqa: E402
from collections import defaultdict
import json
import numpy as np
import copy
import sys
import os
import subprocess
# Add the xG directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'xG')))

from xG_model import xGModel



def process_off_ball_players(run, output_run, xg_model):


    plyrs_xg = {}

    for idx, off_ball_player in enumerate(run['off_ball_teammates']):

    

        plyr_id = off_ball_player['player_id']

        xG_pred_start = get_xG(xg_model, plyr_id, 'start')
        xG_pred_end = get_xG(xg_model, plyr_id, 'end')

        plyrs_xg[plyr_id] = {
            'xG_start': round(xG_pred_start, 3),
            'xG_end': round(xG_pred_end, 3)
        }


    # Update the output run
    for idx, off_ball_player in enumerate(output_run['off_ball_teammates']):
        off_ball_player['metrics']['xG_start'] = plyrs_xg[off_ball_player['player_id']]['xG_start']
        off_ball_player['metrics']['xG_end'] = plyrs_xg[off_ball_player['player_id']]['xG_end']


    return output_run



def get_data_from_s3(match_id):

    try:
        print(f"================================================")
        print(f"Getting data from s3 for match {match_id}")

        check_file1 = f"outputs/{match_id}/merged_tracking_events.jsonl"
        check_file2 = f"outputs/{match_id}/cached_data/match_info.json"

        if os.path.exists(check_file1) and os.path.exists(check_file2):
            print(f"File {check_file1} and {check_file2} already exists")
            return True

        # subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/events.json", "--match_id", str(match_id), "--type", "json"])
        # subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/wy_id2plyr_id.pkl", "--match_id", str(match_id), "--type", "pkl"])
        # subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/wy_id2team_id.pkl", "--match_id", str(match_id), "--type", "pkl"])
        subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/cached_data/match_info.json", "--match_id", str(match_id), "--type", "json"])
        # subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/plyr_lookup_dict.json", "--match_id", str(match_id), "--type", "json"])
        subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/merged_tracking_events.jsonl", "--match_id", str(match_id), "--type", "jsonl"])
        # subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/{match_id}_tracking_data.jsonl", "--match_id", str(match_id), "--type", "jsonl"])
        # subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/possession_intervals.json", "--match_id", str(match_id), "--type", "json"])
        # subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/PPCF_filtered_30x20.npz", "--match_id", str(match_id), "--type", "npz"])
        print(f"================================================")
    except Exception as e:
        print(f"Error getting data from s3 for match {match_id}: {e}")
        return
    
def get_xG(xg_model, plyr_id, type):

    features = xg_model.get_features(type=type, plyr_id=plyr_id, speed_imputation=True)
    xG_pred, _ = xg_model.predict(features)

    return xG_pred

def process_run(run, xg_model):

    

    output_run = copy.deepcopy(run)

    xg_model.load_run(run)

    # Recalculate the xG of the runner

    runner_id = run['player_id']
    xG_pred_start = get_xG(xg_model, runner_id, 'start')
    xG_pred_end = get_xG(xg_model, runner_id, 'end')


    output_run['runner_info']['metrics']['xG_start'] = round(xG_pred_start, 3)

    output_run['runner_info']['metrics']['xG_end'] = round(xG_pred_end, 3)
    

    # Now process the off ball players
    output_run = process_off_ball_players(run, output_run, xg_model)

    

    return output_run





def group_by_match_id(runs):
    grouped_by_match = defaultdict(list)

    for run in runs:
        grouped_by_match[run['match_id']].append(run)

    return grouped_by_match

def main(input_path, output_path):

    output_runs = []

    # Read input file
    with open(input_path, 'r') as f:
        runs = [json.loads(line) for line in f]

    grouped_runs = group_by_match_id(runs)

    for idx, (match_id, runs) in enumerate(grouped_runs.items()):


        try:
            print(f"Processing match {idx + 1} of {len(grouped_runs)}")

            output_flag = get_data_from_s3(match_id)


            log_reg_model = f"stats/xG/models/xg_logistic_calibrated_model.pkl"

            feature_names_path = f"stats/xG/models/xg_logistic_calibrated_feature_names.pkl"


            # Get xG model for the match
            xg_model = xGModel(match_id=match_id, model_path=log_reg_model, feature_names_path=feature_names_path)

            for run in runs:
                output_run = process_run(run, xg_model)
                output_runs.append(output_run)

            # Delete the folder
            if not output_flag:
                subprocess.run(["rm", "-rf", f"outputs/{match_id}"])
                print(f"Deleted folder {match_id}")

            

            

        except Exception as e:
            print(f"Error processing match {match_id}: {e}")
            continue

        

    # Write the output runs to the output file
    with open(output_path, 'w') as f:
        for run in output_runs:
            f.write(json.dumps(run) + '\n')

    print(f"Wrote {len(output_runs)} runs to {output_path}")





if __name__ == "__main__":

    input_path = "stats/obso/datasets/test_filtered_runs.jsonl"
    output_path = "stats/obso/datasets/test_filtered_runs_recalculated_v2.jsonl"

    main(input_path, output_path)
