import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
from ori_config import S3_FETCH_SCRIPT  # noqa: E402
import subprocess
from metrica_pitch_control import MetricaPitchControl
import pandas as pd
import json
import sys
import numpy as np
import pandas as pd
import json
from visual import Visual
import time
import os
import traceback
from format_events import EventDataFormatter
from format_tracking import TrackingDataFormatter


class PitchControl:
    def __init__(self, match_id):
        
        self.match_id = match_id

        self.set_check_paths()
        self.load_necessary_data()

        

        self.tracking_formatter = TrackingDataFormatter(self)

        self.home_tracking_data, self.away_tracking_data = self.tracking_formatter.run()

        self.event_formatter = EventDataFormatter(self) 

        self.events_df = self.event_formatter.run()        

        self.metrica = MetricaPitchControl()
        self.visual = Visual(self)


        self.PPCF_data = []

        # Preprocess tracking data
        self.preprocess_tracking_data()

        # Pitch control output path
        self.pc_passes_output_path = f"outputs/{self.match_id}/PPCF_passes_data.npz"


    
    def set_check_paths(self):
        """
        This method sets the necessary files paths and checks they exist
        """

        self.wy_events_path = f'outputs/{self.match_id}/events.json'
        self.merged_jsonl_path = f'outputs/{self.match_id}/merged_tracking_events.jsonl'
        self.match_tracking_info_path = f'outputs/{self.match_id}/cached_data/match_info.json'
        self.tracking_data_path = f"outputs/{self.match_id}/{self.match_id}_tracking_data.jsonl"
        self.possesion_intervals_path = f"outputs/{self.match_id}/possession_intervals.json"
        self.players_data_path = f"outputs/{self.match_id}/plyr_lookup_dict.json"

        if not os.path.exists(self.wy_events_path):
            raise FileNotFoundError(f"File {self.wy_events_path} does not exist")
        if not os.path.exists(self.merged_jsonl_path):
            raise FileNotFoundError(f"File {self.merged_jsonl_path} does not exist")
        if not os.path.exists(self.match_tracking_info_path):
            raise FileNotFoundError(f"File {self.match_tracking_info_path} does not exist")
        if not os.path.exists(self.tracking_data_path):
            raise FileNotFoundError(f"File {self.tracking_data_path} does not exist")
        if not os.path.exists(self.possesion_intervals_path):
            raise FileNotFoundError(f"File {self.possesion_intervals_path} does not exist")
        if not os.path.exists(self.players_data_path):
            raise FileNotFoundError(f"File {self.players_data_path} does not exist")

        
        
        
    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)
        
    def _load_jsonl(self, path):
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]


    def load_necessary_data(self):
    
        print(f"Loading necessary data for match {self.match_id}")
        
        # Load wyscout events
        self.wy_events = self._load_json(self.wy_events_path)
        
        # Load merged jsonl
        self.merged_jsonl = self._load_jsonl(self.merged_jsonl_path)
        
        # Load match tracking info
        self.match_tracking_info = self._load_json(self.match_tracking_info_path)

        # Load tracking data
        self.tracking_data = self._load_jsonl(self.tracking_data_path)

        # Load possesion intervals
        self.possesion_intervals = self._load_json(self.possesion_intervals_path)
        
            



    def preprocess_tracking_data(self):
        # Convert positions from metrica units to meters
        self.home_tracking_data = self.metrica.to_metric_coordinates(self.home_tracking_data)
        self.away_tracking_data = self.metrica.to_metric_coordinates(self.away_tracking_data)

        # Calculate player velocities
        self.home_tracking_data = self.metrica.calc_player_velocities(self.home_tracking_data,smoothing=True)
        self.away_tracking_data = self.metrica.calc_player_velocities(self.away_tracking_data,smoothing=True)

        # Get pitch control model parameters
        self.params = self.metrica.default_model_params()

        # find goalkeepers for offside calculation
        self.GK_numbers = [self.metrica.find_goalkeeper(self.home_tracking_data),self.metrica.find_goalkeeper(self.away_tracking_data)]


    def get_team_posession(self, video_timestamp):
        """
        This method takes a video timestamp and returns the team in posession at that timestamp
        """
        try:
            for interval in self.possesion_intervals:
                team, start, end, _ = interval
                if float(start) <= video_timestamp <= float(end):
                    return team
        except Exception as e:
            print(f"Error in get_team_posession: {e}")
            traceback.print_exc()
            return None


    def get_home_away_team(self, team_id):
        """
        This method returns the home and away team
        """

        home_team = self.match_tracking_info['home_team_wy_id']
        away_team = self.match_tracking_info['away_team_wy_id']

        if team_id == home_team:
            return "Home"
        elif team_id == away_team:
            return "Away"
        else:
            return None
        


    def process_frame(self, frame_index):

        # Get tracking data for the frame
        home_tracking_data = self.home_tracking_data.loc[frame_index]
        away_tracking_data = self.away_tracking_data.loc[frame_index]

        # Get Video Timestamp
        video_timestamp = home_tracking_data['Time [s]']

        # Get team posession
        team_posession = self.get_team_posession(video_timestamp)

        # Get if team is home or away
        self.team_type = self.get_home_away_team(team_posession)
    
        if team_posession == None:
            print("No team posession found")
            return
        
        # Get pitch control

        start_time = time.time()
        PPCF,xgrid, ygrid  = self.metrica.generate_pitch_control_for_frame(home_tracking_data, away_tracking_data, self.params, self.GK_numbers, possesion_team = self.team_type, field_dimen = (106.,68.,), n_grid_cells_x = 30, offsides=True)
        end_time = time.time()
        print(f"Time taken to process frame {frame_index}: {end_time - start_time:.2f} seconds")

        # Save PPCF data
        self.PPCF_data.append(
            {
                "frame_index": frame_index,
                "PPCF": PPCF,
                "xgrid": xgrid,
                "ygrid": ygrid,
                "team_type": self.team_type
            }
        )
        
        
        
        
    def get_frames_with_passes(self):

        frames_with_passes = []

        # Iterate through events data to find frames with passes
        for index, row in self.events_df.iterrows():
            if row['Type'] == 'PASS':
                frame_index = row['Start Frame']
                frames_with_passes.append(frame_index)


        return frames_with_passes
        
            

    def run_pitch_control_passes(self):

        

        frames_with_passes = self.get_frames_with_passes()

        for idx, frame_index in enumerate(frames_with_passes):
            print(f"Processing frame {idx} of {len(frames_with_passes)}")
            try:
                self.process_frame(frame_index)
            except Exception as e:
                print(f"Error in process_frame: {e}")
                traceback.print_exc()

        # Save PPCF data in Npz format
        np.savez(self.pc_passes_output_path, PPCF_data = self.PPCF_data)


    def run_pc_for_frame(self, frame_index):

        try:    
            self.process_frame(frame_index)
        except Exception as e:
            print(f"Error in process_frame: {e}")
            traceback.print_exc()

        # Get PPCF data
        PPCF_data = self.PPCF_data

        # Get PPCF for the frame
        PPCF = PPCF_data[0]['PPCF']
        xgrid = PPCF_data[0]['xgrid']
        ygrid = PPCF_data[0]['ygrid']


        return PPCF, xgrid, ygrid
    

    def get_pc_at_location(self, frame_index, x, y, PPCF = None,xgrid = None, ygrid = None, field_dimen=(105, 68)):
        """
        Converts (0,0)-(105,68) pitch coordinates to grid-coordinates and returns pitch control.
        """

        if PPCF is None:
            PPCF, xgrid, ygrid = self.run_pc_for_frame(frame_index)

        # Step 1: Convert to centered coordinate system
        x_centered = x - field_dimen[0] / 2  # x - 52.5
        y_centered = y - field_dimen[1] / 2  # y - 34

        # Step 2: Find nearest grid indices
        x_idx = np.argmin(np.abs(xgrid - x_centered))
        y_idx = np.argmin(np.abs(ygrid - y_centered))

        return PPCF[y_idx, x_idx]
    

    def run_pc_for_full_game(self):

        # Make sure PPCF data is empty
        self.PPCF_data = []


        minute_counter = 0

        for idx, frame_index in enumerate(range(len(self.home_tracking_data))):

            try:

                video_time = self.home_tracking_data.iloc[frame_index]['Time [s]']
                time_int = int(video_time)

                if time_int > minute_counter:
                    print(f"processing frame {idx} of {len(self.home_tracking_data)}")


                    self.process_frame(frame_index)
                    minute_counter = time_int

            except Exception as e:
                print(f"Error in process_frame: {e}")
                traceback.print_exc()

        # Save PPCF data in Npz format

        output_path = f"outputs/{self.match_id}/PPCF_filtered_30x20.npz"
        np.savez(output_path, PPCF_data = self.PPCF_data)

def get_data_from_s3(match_id):

    subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/events.json", "--match_id", str(match_id), "--type", "json"])
    subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/wy_id2plyr_id.pkl", "--match_id", str(match_id), "--type", "pkl"])
    subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/wy_id2team_id.pkl", "--match_id", str(match_id), "--type", "pkl"])
    subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/cached_data/match_info.json", "--match_id", str(match_id), "--type", "json"])
    subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/plyr_lookup_dict.json", "--match_id", str(match_id), "--type", "json"])
    subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/merged_tracking_events.jsonl", "--match_id", str(match_id), "--type", "jsonl"])
    subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/possession_intervals.json", "--match_id", str(match_id), "--type", "json"])
    subprocess.run(["python", S3_FETCH_SCRIPT, "--file_path", f"outputs/{match_id}/{match_id}_tracking_data.jsonl", "--match_id", str(match_id), "--type", "jsonl"])
    


if __name__ == "__main__":

    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate pitch control with a given match ID')
    parser.add_argument('match_id', type=int, help='Match ID to process')

    args = parser.parse_args()
    match_id = args.match_id    

    get_data_from_s3(match_id)

    pitch_control = PitchControl(match_id)

    pitch_control.run_pc_for_full_game()


    # MATCHES THAT GAVE ERROR
    error_matches = [622203, 622210, 622256, 622257, 622271, 622295, 622329, 623000, 623006, 623027, 623047, 623058, 623059, 623065, 623175, 623182, 623250, 623259, 623305, 623314, 623356, 623400, 623401, 623407, 638283, 666385]


    errors = {}

    for match_id in error_matches:

        try:
            get_data_from_s3(match_id)
            pitch_control = PitchControl(match_id)
            pitch_control.run_pc_for_full_game()
        except Exception as e:
            print(f"Error in get_data_from_s3: {e}")
            e = traceback.format_exc()
            errors[match_id] = e

    print(errors)
    # PPCF_data = pitch_control.run_pc_for_frame(frame_index)

    # pc = pitch_control.get_pc_at_location(frame_index, 52.5 , 0)

    # print(pc)

    # pc = pitch_control.get_pc_at_location(frame_index, 52.5 , 68)

    # print(pc)

    
    
    # pitch_control.run_pitch_control_passes()
    
    #########################################################
    # HELPFUL COMMANDS
    #########################################################

    # python stats/pitch_control/pitch_control.py 622203

    # zip -r pitch-control-runner.zip pitch_control.py metrica_pitch_control.py format_tracking.py format_events.py visual.py player.py

    


