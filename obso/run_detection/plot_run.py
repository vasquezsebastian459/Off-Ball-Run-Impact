import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import subprocess
import os


import sys
import os
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)


from stats.xReceiver.xgboost_model import XGBoostModel

from stats.xG.xG_model import xGModel




class PlotRun:
    def __init__(self, match_id):
        self.match_id = match_id

        self.xR_model = XGBoostModel(model_path = "stats/xReceiver/models/xgb_model_no_pc.pkl")

        self.xR_model.load_data(match_id=self.match_id)


        log_reg_model = f"stats/xG/models/xg_logistic_calibrated_model.pkl"

        feature_names_path = f"stats/xG/models/xg_logistic_calibrated_feature_names.pkl"


        # Get xG model for the match
        # self.xG_model = xGModel(match_id=match_id, model_path="stats/xG/models/logreg_model.joblib", feature_names_path="stats/xG/models/logreg_feature_names.joblib")
        self.xG_model = xGModel(match_id=match_id, model_path=log_reg_model, feature_names_path=feature_names_path)
        
        
        self.load_data()


        # text paramters
        self.text_size = 12
        self.marker_size = 200


        self.runner_xR = None
        self.runner_xG = None
        self.ppcf = None


    def load_data(self):
        runs_path = f"outputs/{self.match_id}/{self.match_id}_runs.jsonl"
        player_lookup_path = f"outputs/{self.match_id}/plyr_lookup_dict.json"
        # with open(runs_path, "r") as f:
        #     self.runs = [json.loads(line) for line in f]
        with open(player_lookup_path, "r") as f:
            self.player_lookup = json.load(f)

        # Load match info
        match_info_path = f"outputs/{self.match_id}/cached_data/match_info.json"
        with open(match_info_path, "r") as f:
            self.match_info = json.load(f)




    def load_pitch_control(self):
        pitch_control_path = f"outputs/{self.match_id}/PPCF_filtered_30x20.npz"

        data = np.load(pitch_control_path, allow_pickle=True)
        self.pitch_control = data['PPCF_data'].tolist()

    def load_def_team(self, frame_index, type = 'start'):

        tracking_data = f"outputs/{self.match_id}/{self.match_id}_tracking_data.jsonl"

        with open(tracking_data, "r") as f:
            tracking_data = [json.loads(line) for line in f]

        for idx, row in enumerate(tracking_data):

            if idx == 0:
                continue

            tracking_frame = row['frame']

            if tracking_frame == frame_index:
                tracking = row['data']

                for team_id, team_data in tracking.items():
                    if int(team_id) != int(self.team_id):
                        
                        return team_data
                        

    def get_xR(self, plyr_id, type):
        features = self.xR_model.xgb_run_processor.get_features(type=type, plyr_id=plyr_id)
        xP_pred, class_pred  = self.xR_model.predict(features)
        return xP_pred, class_pred
    

    def get_xG(self, plyr_id, type):
        features = self.xG_model.get_features(type=type, plyr_id=plyr_id, speed_imputation=True)
        xP_pred, class_pred  = self.xG_model.predict(features)
        return xP_pred, class_pred
            
    
    def get_pitch_control_frame(self, frame_index):
        
        for frame in self.pitch_control:

            pc_frame_idx = frame.get('frame_index')

            # get distance to the desired frame
            distance = abs(pc_frame_idx - frame_index)

            

            if distance < 10:
                return frame

            
            if frame.get('frame_index') == frame_index:
                return frame
        return None
    
    
    def get_ppcf_at_player_position(self, x_std, y_std, ppcf_frame):
        x_centered = x_std - 105 / 2
        y_centered = y_std - 68 / 2

        PPCF = ppcf_frame['PPCF']
        xgrid = ppcf_frame['xgrid']
        ygrid = ppcf_frame['ygrid']
        control_team = ppcf_frame['team_type']  # 'Home' or 'Away'

        ix = np.argmin(np.abs(xgrid - x_centered))
        iy = np.argmin(np.abs(ygrid - y_centered))

        pc_value = PPCF[iy, ix]

        
        # Invert if the control matrix was calculated for the opposite team
        if control_team != self.team_type:
            pc_value = 1 - pc_value

        return pc_value



    def draw_pitch(self, ax):
        ax.add_patch(patches.Rectangle((0, 0), 105, 68, color='#228B22'))  # pitch green
        ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], color='white', linewidth=2)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    def plot_run(self, run, show_PC = False, show_xR = False, show_xG = False, show_ORI = False):

        # If ORI is True, then show all metrics
        if show_ORI:
            show_xG = True
            show_xR = True
            show_PC = True

        self.run = run

        # Load the run into the xR model

        if show_xR or show_ORI:
            self.xR_model.xgb_run_processor.load_run(run)

        if show_xG or show_ORI:
            self.xG_model.load_run(run)


        # Get start and end frame of run
        self.start_frame = run['start_frame']
        self.end_frame = run['end_frame']

        # Get team

        self.team_id = run['team_id']

        # Get team type
        self.team_type = 'Home' if int(self.match_info['home_team_id']) == int(self.team_id) else 'Away'
        

        
        self.start_def_team = self.load_def_team(self.start_frame, type = 'start')
        self.end_def_team = self.load_def_team(self.end_frame, type = 'end')


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), facecolor='white')

        fig.subplots_adjust(top=0.65)  # to make room for suptitle/text

        # Keep track of team wide metric
        self.team_metric_start = 0
        self.team_metric_end = 0


        for i, ax in enumerate([ax1, ax2]):

            type_team_metric = 0
            
                
            self.draw_pitch(ax)
            frame = run['start_frame'] if i == 0 else run['end_frame']

            # Get Ball Carrier ID
            ball_carrier_id = run['ball_carrier_id_start'] if i == 0 else run['ball_carrier_id_end']

            pc = self.get_pitch_control_frame(frame) if show_PC else None


            time = run["start_timestamp"] if i == 0 else run["end_timestamp"]
            ax.set_title(f"{'Start' if i == 0 else 'End'} of Run - Frame {frame} | Time {time}", fontsize=12, fontweight='bold', color='black')

            runner_pos = run["runner_info"]["start_position"] if i == 0 else run["runner_info"]["end_position"]
            ax.scatter(runner_pos["x"], 68 - runner_pos["y"], marker='*', color="blue", s=self.marker_size, label="Runner", zorder=3)
            # Add runner number
            ax.text(runner_pos["x"] - 0.8, 68 - runner_pos["y"] + 1, self.player_lookup[str(run["player_id"])]["number"], fontsize=self.text_size, color="white", zorder=4)

            # Get PC at runners position
            if pc:
                self.ppcf  = self.get_ppcf_at_player_position(runner_pos["x"], runner_pos["y"], pc)
                # Add to team metric
                type_team_metric += self.ppcf


            # GEt xR Runner
            if show_xR:
                type_run = 'start' if i == 0 else 'end'
                self.runner_xR, _ = self.get_xR(run["player_id"], type_run)

                # Add to team metric
                type_team_metric += self.runner_xR


            # Get xG for Runner
            if show_xG:
                type_run = 'start' if i == 0 else 'end'
                self.runner_xG, _ = self.get_xG(run["player_id"], type_run)

                # Add to team metric
                type_team_metric += self.runner_xG
                

            metric_desired = None
            metric_desired = self.runner_xG if show_xG else metric_desired
            metric_desired = self.runner_xR if metric_desired == None else metric_desired
            metric_desired = self.ppcf if metric_desired == None else metric_desired


            if show_ORI:
                type_team_metric = 0 
                metric_desired = self.ppcf * self.runner_xG * self.runner_xR
                type_team_metric += metric_desired

                ax.text(runner_pos["x"] + 1, 68 - runner_pos["y"] + 2, f"{round(metric_desired, 3)}", fontsize=self.text_size, color="white", zorder=4)

            else:
                # Add text to desired metric
                ax.text(runner_pos["x"] + 1, 68 - runner_pos["y"] + 2, f"{int(metric_desired * 100)}%", fontsize=self.text_size, color="white", zorder=4)


            # Add teammate numbers
            for teammate in run["off_ball_teammates"]:
                pos = teammate["start_position"] if i == 0 else teammate["end_position"]

                # Get TEAMMATE ID

                plyr_id = teammate["player_id"]

                ax.text(pos["x"] - 0.7, 68 - pos["y"] + 1, self.player_lookup[str(teammate["player_id"])]["number"], fontsize=self.text_size, color="white", zorder=4)

                if plyr_id == ball_carrier_id:
                    ax.scatter(pos["x"], 68 - pos["y"], color="blue", s=self.marker_size, zorder=2, marker = "s", label = "Ball Carrier")
                    continue
                else:
                    ax.scatter(pos["x"], 68 - pos["y"], color="blue", s=self.marker_size, zorder=2)
                
                off_ball_metric = None

                self.plyr_xR = None
                self.plyr_xG = None
                self.ppcf = None

                if pc:
                    self.ppcf = self.get_ppcf_at_player_position(pos["x"], pos["y"], pc)    

                    # Add text to the plot
                    # ax.text(pos["x"] + 1, 68 - pos["y"] + 2, f"{int(ppcf*100)}%", fontsize=self.text_size, color="white", zorder=4)

                    # Add to team metric
                    type_team_metric += self.ppcf if not show_ORI else 0

                if show_xR:
                    type_run = 'start' if i == 0 else 'end'
                    self.plyr_xR, _ = self.get_xR(plyr_id, type_run)
                    
                    # ax.text(pos["x"] + 1, 68 - pos["y"] + 2, f"{int(self.plyr_xR * 100)}%", fontsize=self.text_size, color="white", zorder=4)

                    # Add to team metric
                    type_team_metric += self.plyr_xR if not show_ORI else 0

                if show_xG:
                    type_run = 'start' if i == 0 else 'end'
                    self.plyr_xG, _ = self.get_xG(plyr_id, type_run)

                    # ax.text(pos["x"] + 1, 68 - pos["y"] + 2, f"{int(self.plyr_xG * 100)}%", fontsize=self.text_size, color="white", zorder=4)

                    # Add to team metric
                    type_team_metric += self.plyr_xG if not show_ORI else 0

                # Add the metric to the off ball player
                off_ball_metric = self.plyr_xG if show_xG else off_ball_metric
                off_ball_metric = self.plyr_xR if off_ball_metric == None else off_ball_metric
                off_ball_metric = self.ppcf if off_ball_metric == None else off_ball_metric

                if show_ORI:
                    off_ball_metric = self.ppcf * self.plyr_xG * self.plyr_xR
                    type_team_metric += off_ball_metric

                    ax.text(pos["x"] + 1, 68 - pos["y"] + 2, f"{round(off_ball_metric, 3)}", fontsize=self.text_size, color="white", zorder=4)

                else:
                    ax.text(pos["x"] + 1, 68 - pos["y"] + 2, f"{int(off_ball_metric * 100)}%", fontsize=self.text_size, color="white", zorder=4)


            # Add Defenders 

            def_team = self.start_def_team if i == 0 else self.end_def_team

            for defender in def_team:
                
                ax.scatter(defender["x"], 68 - defender["y"], color="red", s=self.marker_size, zorder=2)
                ax.text(defender["x"] - 0.7, 68 - defender["y"] + 1, self.player_lookup[str(defender["id"])]["number"], fontsize=self.text_size, color="white", zorder=4)

            ax.legend(loc="lower right", facecolor="white", edgecolor='black')


            
            # Add to team metric
            if i == 0:
                self.team_metric_start = type_team_metric
            else:
                self.team_metric_end = type_team_metric

        # Get Title based on Annotation Preferences
        fig.suptitle(f"Run {run['run_id']} | Match {run['match_id']}", fontsize=18, fontweight='bold', color='black')

        # Annotation Preference Title
        if show_ORI:
            subtitle = f"ORI SCORE: {self.team_metric_end - self.team_metric_start:.3f}"
        else:
            subtitle = f"% showing: {'PC' if show_PC else ''} {'xR' if show_xR else ''} {'xG' if show_xG else ''}"

        fig.text(0.5, 0.90, subtitle, fontsize=14, ha='center', color='black')

        # Add team metric
        if show_ORI:
            fig.text(0.5, 0.85, f"Teamwide OBSO at Start: {self.team_metric_start:.3f} | Teamwide OBSO at End: {self.team_metric_end:.3f}", fontsize=14, ha='center', color='black')
        else:
            fig.text(0.5, 0.85, f"Agg Team Metric at Start: {self.team_metric_start:.2f}% | Agg Team Metric at End: {self.team_metric_end:.2f}%", fontsize=14, ha='center', color='black')

        return fig



def get_data_from_s3(match_id):
    print(f"================================================")
    print(f"Getting data from s3 for match {match_id}")

    check_path = f"outputs/{match_id}/PPCF_filtered_30x20.npz"

    if os.path.exists(check_path):
        print(f"PPCF already exists for match {match_id}")
        return

    subprocess.run(["python", "/Users/svasquez/AWS_PC/s3_get_file.py", "--file_path", f"outputs/{match_id}/events.json", "--match_id", str(match_id), "--type", "json"])
    subprocess.run(["python", "/Users/svasquez/AWS_PC/s3_get_file.py", "--file_path", f"outputs/{match_id}/wy_id2plyr_id.pkl", "--match_id", str(match_id), "--type", "pkl"])
    subprocess.run(["python", "/Users/svasquez/AWS_PC/s3_get_file.py", "--file_path", f"outputs/{match_id}/wy_id2team_id.pkl", "--match_id", str(match_id), "--type", "pkl"])
    subprocess.run(["python", "/Users/svasquez/AWS_PC/s3_get_file.py", "--file_path", f"outputs/{match_id}/cached_data/match_info.json", "--match_id", str(match_id), "--type", "json"])
    subprocess.run(["python", "/Users/svasquez/AWS_PC/s3_get_file.py", "--file_path", f"outputs/{match_id}/plyr_lookup_dict.json", "--match_id", str(match_id), "--type", "json"])
    # subprocess.run(["python", "/Users/svasquez/AWS_PC/s3_get_file.py", "--file_path", f"outputs/{match_id}/merged_tracking_events.jsonl", "--match_id", str(match_id), "--type", "jsonl"])
    subprocess.run(["python", "/Users/svasquez/AWS_PC/s3_get_file.py", "--file_path", f"outputs/{match_id}/{match_id}_tracking_data.jsonl", "--match_id", str(match_id), "--type", "jsonl"])
    subprocess.run(["python", "/Users/svasquez/AWS_PC/s3_get_file.py", "--file_path", f"outputs/{match_id}/possession_intervals.json", "--match_id", str(match_id), "--type", "json"])
    subprocess.run(["python", "/Users/svasquez/AWS_PC/s3_get_file.py", "--file_path", f"outputs/{match_id}/PPCF_filtered_30x20.npz", "--match_id", str(match_id), "--type", "npz"])
    print(f"================================================")

def make_run_dir(match_id, run_id):
    img_dir = f"/Users/svasquez/Desktop/Capstone Project/Images"

    # Make sure there is a folder for the match
    match_dir = f"{img_dir}/{match_id}"
    if not os.path.exists(match_dir):
        os.makedirs(match_dir)

    # Make sure there is a folder for the run
    run_dir = f"{match_dir}/{run_id}"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    return run_dir


def plot_run(match_id, run, metric_desired):

    pr = PlotRun(match_id)

    print(f"LOADING PITCH CONTROL ...")
    pr.load_pitch_control()

    print(f"PLOTTING RUN ...")
    show_PC = False
    show_xR = False
    show_xG = False
    show_ORI = False
    if metric_desired == 'xG':
        show_xG = True
    elif metric_desired == 'xR':
        show_xR = True
    
    elif metric_desired == 'PC':
        show_PC = True
    elif metric_desired == 'ORI':
        show_ORI = True
    
    fig = pr.plot_run(run, show_PC = show_PC, show_xR = show_xR, show_xG = show_xG, show_ORI = show_ORI)

    return fig


def save_fig(fig, run_id, metric_desired):

    run_dir = make_run_dir(match_id, run_id)

    # Save figure
    run_img_path = f"{run_dir}/{run_id}_{metric_desired}.png"

    # Finalize layout and spacing
    fig.tight_layout()
    fig.subplots_adjust(top=0.75)  # Adjust to leave room for the suptitle and other texts

    # Save with consistent resolution and no cropping
    fig.savefig(run_img_path, dpi=300)

    print(f"RUN {run_id} SAVED TO {run_img_path}")

if __name__ == "__main__":
    
    
    run_id = "623337_160"
    run_id = "622163_254" # WORKS
    run_id = "622209_171" # WORKS
    # run_id = "622266_5"
    run_id = "622094_96"
    run_id = "622209_111" # MBAPPE's WORST RUN according to OBSO
    run_id = "622278_293" # MBAPPE'S BEST RUN
    run_id = "622311_14" # Julian Alvarez BEST RUN
    # run_id = "622092_407"  # Not effecrtive run

    match_id = run_id.split("_")[0]

    get_data_from_s3(match_id)

    runs_path = f"stats/obso/datasets/test_filtered_runs.jsonl"

    runs = []
    with open(runs_path, "r") as f:
        for line in f:
            runs.append(json.loads(line))


    

    print(f"GETTING RUN {run_id} ...")
    run = None
    for train_run in runs:

        if train_run['run_id'] == run_id:
            run = train_run
    
    if run is None:
        print(f"RUN {run_id} NOT FOUND")
        exit()
    
    print(f"RUN {run_id} FOUND")

    metric_desired = 'xG'
    fig = plot_run(match_id, run, metric_desired)
    save_fig(fig, run_id, metric_desired)

    metric_desired = 'xR'
    fig = plot_run(match_id, run, metric_desired)
    save_fig(fig, run_id, metric_desired)

    metric_desired = 'PC'
    fig = plot_run(match_id, run, metric_desired)
    save_fig(fig, run_id, metric_desired)


    metric_desired = 'ORI'
    fig = plot_run(match_id, run, metric_desired)
    save_fig(fig, run_id, metric_desired)



    

