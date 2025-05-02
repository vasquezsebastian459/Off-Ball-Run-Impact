import json
import pandas as pd


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pitch_control import PitchControl

class TrackingDataFormatter:

    def __init__(self, pitch_control: 'PitchControl'):

        self.pitch_control = pitch_control

        self.match_id = self.pitch_control.match_id

        self.tracking_data = self.pitch_control.tracking_data

        self.meta_data = self.get_meta_data()

        # Get players data from meta data
        self.players_data = self.meta_data['players_data']

        self.match_tracking_info = self.pitch_control.match_tracking_info

        self.tracking_home = []
        self.tracking_away = []

        self.output_dir = f"outputs/{self.match_id}"


    def get_meta_data(self):

        meta_data = None
        for row in self.tracking_data:
            if 'match_data' in row:
                meta_data = row
                break
        
        return meta_data

    def meters_to_normalized_metrica(self, x_m, y_m, field_length=105, field_width=68):
        x_norm = x_m / field_length

        # Flip the y-axis
        y_norm = 1 - (y_m / field_width)
        return x_norm, y_norm

    def get_row_data(self, row, type = 'Home'):

        # Get the basic data
        row_time = row['Videotimestamp']
        row_frame = row['frame']
        row_period = row['period']
        
        

        team_id = str(self.match_tracking_info[f'{type.lower()}_team_id'])

        output_row = {
                'Frame': row_frame,
                'Period': row_period,
                'Time [s]': row_time
            }

        # Get player coords
        row_data = row['data']
        coords = row_data[team_id]

        for player in coords:
            plyr_id = player['id']
            player_data = self.players_data[team_id][str(plyr_id)]
            plyr_num = player_data['number'] # SHIRT NUMBER

            # Add it to the tracking home list

            x_norm, y_norm = self.meters_to_normalized_metrica(player['x'], player['y'])

            output_row[f'{type}_{plyr_num}_x'] = x_norm
            output_row[f'{type}_{plyr_num}_y'] = y_norm
        
        # Add the ball coords at the end

        if row['ball'][0] != None:
            ball_x, ball_y = self.meters_to_normalized_metrica(row['ball'][0], row['ball'][1])
        
        else:
            ball_x = None
            ball_y = None
        

        output_row['ball_x'] = ball_x
        output_row['ball_y'] = ball_y

        return output_row

    def format_tracking_data(self):
        
        print(f"Formatting tracking data...")
        for row in self.tracking_data:
            
            # Skip the meta data
            if 'match_data' in row:
                continue

            home_row = self.get_row_data(row, type = 'Home')
            away_row = self.get_row_data(row, type = 'Away')

            # Add it to the tracking home list
            self.tracking_home.append(home_row)

            # Add it to the tracking away list
            self.tracking_away.append(away_row)

        # Turn Both dfs into pandas
        self.tracking_home = pd.DataFrame(self.tracking_home)
        self.tracking_away = pd.DataFrame(self.tracking_away)

        return self.tracking_home, self.tracking_away

    def save_tracking_data(self):

        # Save the tracking data
        self.tracking_home.to_csv(f'{self.output_dir}/tracking_home_converted.csv', index=False)
        self.tracking_away.to_csv(f'{self.output_dir}/tracking_away_converted.csv', index=False)

        print(f"Tracking data saved to:\n- {self.output_dir}/tracking_home_converted.csv \n- {self.output_dir}/tracking_away_converted.csv")

    def run(self):

        self.format_tracking_data()

        # Save the tracking data
        self.save_tracking_data()

        return self.tracking_home, self.tracking_away




if __name__ == "__main__":

    match_id = 666385

    tracking_data_path = f"/Users/svasquez/driblab/dl-vision/outputs/{match_id}/666385_tracking_data.jsonl"

    # Load tracking data
    tracking_data = []
    with open(tracking_data_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                meta_data = json.loads(line)
            else:
                tracking_data.append(json.loads(line))

    match_tracking_info = {
        'match_id': 666385, 
        'match_wy_id': 5589087,
        'home_team_id': 3659, 
        'home_team_wy_id': 2446, 
        'away_team_id': 3651, 
        'away_team_wy_id': 2444, 
        'side_attack': False,
        'season_id': 818378}


    formatter = TrackingDataFormatter(tracking_data, meta_data, match_tracking_info)

    formatter.run()
