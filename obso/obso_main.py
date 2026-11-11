import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
from ori_config import MATCH_LISTS_DIR, REPO_ROOT, S3_FETCH_SCRIPT  # noqa: E402
import traceback
import pandas as pd
from run_detection.run_detection import detect_runs
import subprocess
import sys
import os
sys.path.append(REPO_ROOT)
from xReceiver.xgboost_model import XGBoostModel
from run_detection.plot_run import PlotRun

from xG.xG_model import xGModel

from json_encoder import NumpyEncoder

import json


def get_xR(model : XGBoostModel, plyr_id, type):
    features = model.xgb_run_processor.get_features(type=type, plyr_id=plyr_id)
    xP_pred, class_pred  = model.predict(features)
    return xP_pred, class_pred

def add_xR(run, xR_model:XGBoostModel):

    
    xR_model.xgb_run_processor.load_run(run)

    # get the xR of the runner

    runner_id = run['player_id']

    runner_xR_start, _ = get_xR(xR_model, runner_id, 'start')
    runner_xR_end, _ = get_xR(xR_model, runner_id, 'end')

    # Add it to the run dict
    run['runner_info']['metrics']['xR_start'] = round(runner_xR_start, 3)
    run['runner_info']['metrics']['xR_end'] = round(runner_xR_end, 3)

    # Add the xR to the rest of the off ball players

    off_ball_plyrs = run['off_ball_teammates']

    for plyr in off_ball_plyrs:
        plyr_xR_start, _ = get_xR(xR_model, plyr['player_id'], 'start')
        plyr_xR_end, _ = get_xR(xR_model, plyr['player_id'], 'end')

        plyr['metrics']['xR_start'] = round(plyr_xR_start, 3)
        plyr['metrics']['xR_end'] = round(plyr_xR_end, 3)

    return run


def get_PC_at_position(frame_index, x_coord, y_coord, plot_run:PlotRun):

    # Get pitch control
    pc = plot_run.get_pitch_control_frame(frame_index)

    # Get PC at runners position
    ppcf = plot_run.get_ppcf_at_player_position(x_coord, y_coord, pc)

    return ppcf
    
    
def add_PC_run(run, plot_run:PlotRun, xR_model:XGBoostModel):


    team_id = run['team_id']    
    plot_run.team_type = 'Home' if int(xR_model.match_info['home_team_id']) == int(team_id) else 'Away'

    # get the runner PC
    start_frame = run['start_frame']
    end_frame = run['end_frame']

    start_runner_x, start_runner_y = run['runner_info']['start_position']['x'], run['runner_info']['start_position']['y']
    end_runner_x, end_runner_y = run['runner_info']['end_position']['x'], run['runner_info']['end_position']['y']

    start_ppcf = get_PC_at_position(start_frame, start_runner_x, start_runner_y, plot_run)
    end_ppcf = get_PC_at_position(end_frame, end_runner_x, end_runner_y, plot_run)

    # Add the PC to the run
    run['runner_info']['metrics']['PC_start'] = round(start_ppcf, 3)
    run['runner_info']['metrics']['PC_end'] = round(end_ppcf, 3)

    # get the PC of the off ball players

    off_ball_plyrs = run['off_ball_teammates']

    for plyr in off_ball_plyrs:
        start_plyr_x, start_plyr_y = plyr['start_position']['x'], plyr['start_position']['y']
        end_plyr_x, end_plyr_y = plyr['end_position']['x'], plyr['end_position']['y']

        start_plyr_ppcf = get_PC_at_position(start_frame, start_plyr_x, start_plyr_y, plot_run)
        end_plyr_ppcf = get_PC_at_position(end_frame, end_plyr_x, end_plyr_y, plot_run)

        plyr['metrics']['PC_start'] = round(start_plyr_ppcf, 3)
        plyr['metrics']['PC_end'] = round(end_plyr_ppcf, 3)

    return run


def get_xG(model : xGModel, plyr_id, type):
    features = model.get_features(type=type, plyr_id=plyr_id)
    xP_pred, class_pred  = model.predict(features)
    return xP_pred, class_pred

def add_xG(run, xG_model:xGModel):

    xG_model.load_run(run)    

    # get the xG of the runner

    runner_id = run['player_id']

    runner_xG_start, _ = get_xG(xG_model, runner_id, 'start')
    runner_xG_end, _ = get_xG(xG_model, runner_id, 'end')

    # Add the xG to the run
    run['runner_info']['metrics']['xG_start'] = round(runner_xG_start, 3)
    run['runner_info']['metrics']['xG_end'] = round(runner_xG_end, 3)

    

    # Add the xG to the rest of the off ball players

    off_ball_plyrs = run['off_ball_teammates']

    for plyr in off_ball_plyrs:
        plyr_xG_start, _ = get_xG(xG_model, plyr['player_id'], 'start')
        plyr_xG_end, _ = get_xG(xG_model, plyr['player_id'], 'end')

        plyr['metrics']['xG_start'] = round(plyr_xG_start, 3)
        plyr['metrics']['xG_end'] = round(plyr_xG_end, 3)

    return run

def add_metrics(runs, xR_model:XGBoostModel, plot_run:PlotRun, xG_model:xGModel):

    runs_with_metrics = []

    for idx, run in enumerate(runs):
        print(f"\rAdding Metrics to run {idx + 1} of {len(runs)}", end='', flush=True)

        try:
            run = add_xR(run, xR_model)
            run = add_PC_run(run, plot_run, xR_model)
            run = add_xG(run, xG_model)
            runs_with_metrics.append(run)
        except Exception as e:
            e = traceback.format_exc()
            print(f"Error adding metrics to run {idx + 1} of {len(runs)}: {e}")
            continue


    return runs_with_metrics




def clean_runs(runs):

    clean_runs = []

    for run in runs:

        duration = run['end_timestamp'] - run['start_timestamp']

        # Get if the ball carrier is the player_id
        ball_carrier_id_start = run.get('ball_carrier_id_start')
        ball_carrier_id_end = run.get('ball_carrier_id_end')

        plyr_id = run.get('player_id')

        # Determine if the run is off ball
        off_ball:bool = False if plyr_id == ball_carrier_id_start else True

        # Check if the ball carrier at the end is the pass giver
        received_ball:bool = True if plyr_id == ball_carrier_id_end else False

        if received_ball:
            run['ball_carrier_id_end'] = run.get('ball_carrier_id_start')

        if run.get('ball_carrier_id_start') is not None and run.get('ball_carrier_id_end') is not None and duration < 20 and off_ball:
            clean_runs.append(run)

    return clean_runs



def get_data_from_s3(match_id):

    try:
        print(f"================================================")
        print(f"Getting data from s3 for match {match_id}")

        check_file = f"outputs/{match_id}/PPCF_filtered_30x20.npz"

        if os.path.exists(check_file):
            print(f"File {check_file} already exists")
            return

        subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/events.json", "--match_id", str(match_id), "--type", "json"])
        subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/wy_id2plyr_id.pkl", "--match_id", str(match_id), "--type", "pkl"])
        subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/wy_id2team_id.pkl", "--match_id", str(match_id), "--type", "pkl"])
        subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/cached_data/match_info.json", "--match_id", str(match_id), "--type", "json"])
        subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/plyr_lookup_dict.json", "--match_id", str(match_id), "--type", "json"])
        subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/merged_tracking_events.jsonl", "--match_id", str(match_id), "--type", "jsonl"])
        subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/{match_id}_tracking_data.jsonl", "--match_id", str(match_id), "--type", "jsonl"])
        subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/possession_intervals.json", "--match_id", str(match_id), "--type", "json"])
        subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/PPCF_filtered_30x20.npz", "--match_id", str(match_id), "--type", "npz"])
        print(f"================================================")
    except Exception as e:
        print(f"Error getting data from s3 for match {match_id}: {e}")
        return


def main(match_id):

    print(f"Processing match {match_id}")

    get_data_from_s3(match_id)



    # Detect runs


    runs = detect_runs(match_id)

    print(f"Detected {len(runs)} runs")

    # Clean runs

    cleaned_runs = clean_runs(runs)

    print(f"Cleaned {len(cleaned_runs)} runs")
    # load Models

    # Expected Receiver Model
    xR_model = XGBoostModel(model_path="stats/xReceiver/models/xgb_model_no_pc.pkl")
    xR_model.load_data(match_id)

    # Pitch Control Model
    plot_run = PlotRun(match_id)

    plot_run.load_pitch_control()

    # xG Model

    xG_model = xGModel(match_id=match_id, model_path="stats/xG/models/logreg_model.joblib", feature_names_path="stats/xG/models/logreg_feature_names.joblib")


    # Add Metrics to runs
    runs_with_metrics = add_metrics(cleaned_runs, xR_model, plot_run, xG_model)

    # Save runs with metrics

    # Save runs to jsonl file
    output_path = f"outputs/{match_id}/runs_with_metrics.jsonl"
    with open(output_path, 'w') as f:
        for run in runs_with_metrics:
            f.write(json.dumps(run, cls=NumpyEncoder) + '\n')


    # Add it to the full dataset
    full_dataset_path = f"stats/obso/datasets/training_run_v2.jsonl"

    full_dataset = []
    with open(full_dataset_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                full_dataset.append(json.loads(line))

    full_dataset.extend(runs_with_metrics)

    with open(full_dataset_path, 'w') as f:
        for run in full_dataset:
            f.write(json.dumps(run, cls=NumpyEncoder) + '\n')

    # Delete folder
    # subprocess.run(["rm", "-rf", f"outputs/{match_id}"])












if __name__ == "__main__":

    # Get the match_ids already processed

    training_set_path = f"stats/obso/datasets/training_run_v2.jsonl"
    training_set_path = f"stats/obso/datasets/testing_runs.jsonl"

    training_set = set()
    with open(training_set_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                training_set.add(json.loads(line)['match_id'])


    print(f"LENGTH OF TRAINING SET: {len(training_set)}")
    print(training_set)

    training_set = list(training_set)


    df_path = f"{MATCH_LISTS_DIR}/loaded_matches.csv"

    df = pd.read_csv(df_path)

    training_matches = df[df['training_set'] == False]['match_id'].to_list()

    print(training_matches)
    print(len(training_matches))


    ignore_matches = [622203,622210 ]


    final_matches = []
    for match_id in training_matches:
        if match_id in training_set:
            # print(f"MATCH {match_id} IS IN TRAINING SET")
            pass

        elif match_id in ignore_matches:
            print(f"MATCH {match_id} IS IN IGNORE SET")
            pass
        else:
            print(f"MATCH {match_id} IS NOT IN TRAINING SET")
            final_matches.append(match_id)

    print(final_matches)
    print(len(final_matches))


    for match_id in final_matches:
        main(match_id)

    exit()
    

    # Turn set into list
    ignore_matches = list(training_set)




    df_path = f"{MATCH_LISTS_DIR}/loaded_matches.csv"

    df = pd.read_csv(df_path)

    training_matches = df[df['training_set'] == True]['match_id'].to_list()

    print(training_matches)
    print(len(training_matches))

    # Remove matches that are already processed
    training_matches = [match_id for match_id in training_matches if match_id not in ignore_matches]

    print(training_matches)
    print(len(training_matches))

    


    errors = []

    for idx, match_id in enumerate(training_matches):

        try:


            print(f"Processing match {idx + 1} of {len(training_matches)} -> {match_id}")
            main(match_id)

        except Exception as e:
            

            e = traceback.format_exc()

            print(f"Error processing match {idx + 1} of {len(training_matches)}: {e} -> {match_id}")
            errors.append({
                'match_id': match_id,
                'error': e
            })
            continue
