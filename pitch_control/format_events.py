
import json
import pandas as pd
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pitch_control import PitchControl


class EventDataFormatter:

    def __init__(self, pitch_control: 'PitchControl'):

        self.pitch_control = pitch_control

        self.match_id = self.pitch_control.match_id

        self.wy_events = self.pitch_control.wy_events

        self.merged_jsonl = self.pitch_control.merged_jsonl

        self.match_tracking_info = self.pitch_control.match_tracking_info

        self.events_df = []

        self.event_row = {}

        self.frame_info_mapping = {}

        self.home_team_id = self.match_tracking_info['home_team_wy_id'] # Wyscout ID
        self.away_team_id = self.match_tracking_info['away_team_wy_id'] # Wyscout ID
        self.side_attack = self.match_tracking_info['side_attack'] # True if home team is attacking left first Half

        # Output file 
        self.output_file = f'outputs/{self.match_id}/events_converted.csv'


    
    def init_event_row(self):

        self.event_row = {
            'Team': None,
            'Type': None,
            'Subtype': None,
            'Period': None,
            'Start Frame': None,
            'Start Time [s]': None,
            'End Frame': None,
            'End Time [s]': None,
            'From': None,
            'To': None,
            'Start X': np.nan,
            'Start Y': np.nan,
            'End X': np.nan,
            'End Y': np.nan,
            'Event ID': None
        }

    def normalize_coords_wyscout(self, x, y):
        """
        Convert Wyscout 0–100 coordinate system to normalized 0–1 system,
        where (0,0) is top-left and (1,1) is bottom-right.
        """
        x_norm = x / 100
        y_norm = (y / 100)  # flip Y because Wyscout has (0,0) at bottom-left
        return x_norm, y_norm


    def flip_coords_if_needed(self, x_coord:float, y_coord:float, team_id:int):

        # Turn period into int

        period = self.event_row['Period']

        # Determine who is attacking L ➜ R in this half
        try:
            # Try to divide period by 2 to determine if even/odd
            if period % 2 == 0:  # Period 2 or 4
                left_to_right_team = self.away_team_id if not self.side_attack else self.home_team_id
            else:  # Period 1 or 3
                left_to_right_team = self.home_team_id if not self.side_attack else self.away_team_id
        except:
            # If division fails, assume period 1
            print("Division failed")
            left_to_right_team = self.home_team_id if not self.side_attack else self.away_team_id

        # Normalize coords
        x_coord, y_coord = self.normalize_coords_wyscout(x_coord, y_coord)

        # Flip if team is not attacking left ➜ right in this half
        if team_id != left_to_right_team:
            x_coord = 1 - x_coord
            y_coord = 1 - y_coord

        return x_coord, y_coord
    
    def get_basic_info(self, event):

        self.event_row['Type'] = event['type']['primary'].upper()
        self.event_row['Subtype'] = event['type']['secondary'][0].upper() if event['type']['secondary'] else ''
        self.event_row['Period'] = int(event['matchPeriod'][0])


        self.event_row['From'] = event['player']['name']
        self.event_row['To'] = event['pass']['recipient']['name'] if event['pass'] else ''
        self.event_row['Event ID'] = event['id']

    
    def get_team(self, event):

        team = None

        if event['team']['id'] == self.home_team_id:
            team = 'Home'
        elif event['team']['id'] == self.away_team_id:
            team = 'Away'
        
        return team

    def process_event(self, event):

        self.init_event_row()

        self.get_basic_info(event)

        self.event_row['Team'] = self.get_team(event)

        # Get the Start Coords
        if event['location']:
            self.event_row['Start X'], self.event_row['Start Y'] = self.flip_coords_if_needed(event['location']['x'], event['location']['y'], event['team']['id'])

        # Get the End Coords
        if event['pass']:
            self.event_row['End X'], self.event_row['End Y'] = self.flip_coords_if_needed(event['pass']['endLocation']['x'], event['pass']['endLocation']['y'], event['team']['id'])

        return self.event_row
    
    def get_frame_info_mapping(self):

        for row in self.merged_jsonl:
            if row.get('id'):
                
                # If first sequence, add to frame_info_mapping

                if row['event_seq_num'] == 1:
                    self.frame_info_mapping[row['id']] = {
                        'start_frame': row['frame'],
                        'start_timestamp': row['Videotimestamp']
                    }
                elif row['is_last'] == True:
                    self.frame_info_mapping[row['id']]['end_frame'] = row['frame']
                    self.frame_info_mapping[row['id']]['end_timestamp'] = row['Videotimestamp']
                    
    
    def add_frame_data(self):

        for event in self.events_df:

            event_id = event['Event ID']

            frame_info = self.frame_info_mapping.get(event_id)

            if frame_info:                
                event['Start Frame'] = frame_info.get('start_frame')
                event['Start Time [s]'] = frame_info.get('start_timestamp')
                event['End Frame'] = frame_info.get('end_frame')
                event['End Time [s]'] = frame_info.get('end_timestamp')

            else:
                print(f"No frame info found for event {event_id}")

            

    def run(self):

        print(f"Formatting events for match {self.match_id}")
        
        # Process each event in wy_events
        for event in self.wy_events:
            self.process_event(event)
            self.events_df.append(self.event_row)

        # Post Process the events to add frame info

        self.get_frame_info_mapping()

        # After making the frame info mapping, add the frame info to the events_df

        self.add_frame_data()

        # Save to CSV
        df = pd.DataFrame(self.events_df)
        df.to_csv(self.output_file, index=True)

        return df


# Example usage
if __name__ == "__main__":

    match_id = 666385

    events_file = f'/Users/svasquez/driblab/dl-vision/outputs/{match_id}/events.json'
    merged_jsonl = f'/Users/svasquez/driblab/dl-vision/outputs/{match_id}/merged_tracking_events.jsonl'

    # Load your JSON data
    with open(events_file, 'r') as f:
        wy_events = json.load(f)


    # Load merged jsonl
    merged_data = []
    with open(merged_jsonl, 'r') as f:
        for line in f:
            merged_data.append(json.loads(line))


    match_tracking_info = {
        'match_id': 666385, 
        'match_wy_id': 5589087,
        'home_team_id': 3659, 
        'home_team_wy_id': 2446, 
        'away_team_id': 3651, 
        'away_team_wy_id': 2444, 
        'side_attack': False,
        'season_id': 818378}
    
    formatter = EventDataFormatter(wy_events, merged_data, match_tracking_info)

    formatter.run()
