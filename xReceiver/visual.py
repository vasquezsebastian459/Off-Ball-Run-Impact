
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from xgboost_model import XGBoostModel

import matplotlib.pyplot as plt

class Visualizer:

    def __init__(self, model: 'XGBoostModel'):
        self.model = model

    def set_frame(self, row_num: int):
        self.row_num = row_num

        self.row = self.model.merged_jsonl[row_num]

        self.giver_id = self.row.get('pass', {}).get('player', {}).get('id')
        self.receiver_id = self.row.get('pass', {}).get('receiver', {}).get('id')

        self.giver_coord = self.model.plyr_lookup_dict.get(self.giver_id)
        self.receiver_coord = self.model.plyr_lookup_dict.get(self.receiver_id)


    def set_predictions(self, predicitons_dict: dict):
        self.predictions_dict = predicitons_dict


    def plot_run(self):


        print(self.model.xgb_run_processor.run)

        exit()


        # Get start and end frame of run
        self.start_frame = self.run['start_frame']
        self.end_frame = self.run['end_frame']

        # Get team

        self.team_id = self.run['team_id']

        # Get team type
        self.team_type = 'Home' if int(self.model.match_info['home_team_id']) == int(self.team_id) else 'Away'

        




        

        

        # if def_team:
        #     self.start_def_team = self.run[(self.start_frame, type = 'start')
        #     self.end_def_team = self.load_def_team(self.end_frame, type = 'end')


    




        fig = plt.figure(figsize=(24, 8), facecolor='white')  # outer background white

        ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.8])  # left
        ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.8])  # right

        for i, ax in enumerate([ax1, ax2]):
            
                
            self.draw_pitch(ax)
            frame = run['start_frame'] if i == 0 else run['end_frame']

            # Get Ball Carrier ID
            ball_carrier_id = run['ball_carrier_id_start'] if i == 0 else run['ball_carrier_id_end']


            

            time = run["start_timestamp"] if i == 0 else run["end_timestamp"]
            ax.set_title(f"{'Start' if i == 0 else 'End'} of Run - Frame {frame} | Time {time}", fontsize=15, fontweight='bold', color='black')

            runner_pos = run["runner_info"]["start_position"] if i == 0 else run["runner_info"]["end_position"]
            ax.scatter(runner_pos["x"], 68 - runner_pos["y"], marker='*', color="blue", s=120, label="Runner", zorder=3)
            # Add runner number
            ax.text(runner_pos["x"] - 0.8, 68 - runner_pos["y"] + 0.5, player_lookup[str(run["player_id"])]["number"], fontsize=8, color="white", zorder=4)

            # Get PC at runners position
            if pc:
                ppcf = self.get_ppcf_at_player_position(runner_pos["x"], runner_pos["y"], pc)
                
                # Add text to the plot
                ax.text(runner_pos["x"] + 1, 68 - runner_pos["y"] + 0.5, f"{ppcf:.2f}", fontsize=8, color="white", zorder=4)

            # Add teammate numbers
            for teammate in run["off_ball_teammates"]:
                pos = teammate["start_position"] if i == 0 else teammate["end_position"]

                # Get ID

                plyr_id = teammate["player_id"]

                if plyr_id == ball_carrier_id:
                    ax.scatter(pos["x"], 68 - pos["y"], color="blue", s=60, zorder=2, marker = "s", label = "Ball Carrier")
                else:
                    ax.scatter(pos["x"], 68 - pos["y"], color="blue", s=60, zorder=2)
                
                ax.text(pos["x"] + 1, 68 - pos["y"] + 1, player_lookup[str(teammate["player_id"])]["number"], fontsize=8, color="white", zorder=4)

                if pc:
                    ppcf = self.get_ppcf_at_player_position(pos["x"], pos["y"], pc)
                    
                    
                    # Add text to the plot
                    ax.text(pos["x"] + 1, 68 - pos["y"] + 2, f"{ppcf:.2f}", fontsize=8, color="white", zorder=4)


            # Add Defenders if asked
            if def_team:

                def_team = self.start_def_team if i == 0 else self.end_def_team

                for defender in def_team:
                    
                    ax.scatter(defender["x"], 68 - defender["y"], color="red", s=60, zorder=2)
                    ax.text(defender["x"] + 1, 68 - defender["y"] + 1, player_lookup[str(defender["id"])]["number"], fontsize=8, color="white", zorder=4)

            ax.legend(loc="lower right", facecolor="white", edgecolor='black')

        fig.suptitle(f"Run {run['run_id']} | Match {run['match_id']}", fontsize=18, fontweight='bold', color='black')
        return fig


    def plot_frame(self, predictions_dict: dict = None):

        frame_data = self.row
        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw pitch
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.set_aspect('equal')
        ax.set_facecolor("green")
        ax.set_title(f"Match Frame: {frame_data.get('frame')} - Time: {frame_data.get('videoTimestamp', 0):.2f}s")

        # Get the plyr_id of the passer and receiver

        passer_id = frame_data.get('player', {}).get('id')
        recipient_id = frame_data.get('pass', {}).get('recipient', {}).get('id')

        # Turn to dribab_id 

        passer_id = self.model.wy_id2plyr_id.get(passer_id)
        recipient_id = self.model.wy_id2plyr_id.get(recipient_id)

        passer_coord = None
        recipient_coord = None

        # Team colors
        team_colors = {
            '3651': 'red',     # Home team
            '3659': 'blue'     # Away team
        }



        for team_id, players in frame_data['data'].items():
            for player in players:

            
                

                x, y = player['x'], player['y']

                if int(player['id']) == int(passer_id):

                    passer_coord = (x, y)
                    ax.plot(x, y, 'x', color=team_colors.get(team_id), markersize=10)
                    ax.text(x, y + 1, str(player['id']), fontsize=6, ha='center', color='white')

                elif int(player['id']) == int(recipient_id):
                    recipient_coord = (x, y)
                    ax.plot(x, y, '>', color=team_colors.get(team_id), markersize=10)
                    

                    
                    prediction = round(predictions_dict.get(player['id']).get('xP_pred'), 2)
                    ax.text(x, y + 1, str(prediction), fontsize=6, ha='center', color='white')

                
                    # ax.text(x, y + 1, str(player['id']), fontsize=6, ha='center', color='white')
                            
                else:
                    
                    ax.plot(x, y, 'o', color=team_colors.get(team_id), markersize=10)
                    

                    plyr_pred = predictions_dict.get(player['id'])

                    if plyr_pred:
                        prediction = round(plyr_pred.get('xP_pred'), 2)

                    else:
                        prediction = ""

                    
                    ax.text(x, y + 1, str(prediction), fontsize=6, ha='center', color='white')
                    

                    

        # Plot the ball if available
        ball = frame_data.get('ball')
        if ball and ball[0] is not None and ball[1] is not None:
            ax.plot(ball[0], ball[1], 'o', color='black', markersize=8)
            ax.text(ball[0], ball[1] + 1, "Ball", fontsize=8, color='black')    
    
        # Plot a line between the passer and recipient
        if passer_coord and recipient_coord:
            ax.plot([passer_coord[0], recipient_coord[0]], [passer_coord[1], recipient_coord[1]], 'k-', linewidth=2)

        plt.grid(False)
        plt.xlabel("Pitch X")
        plt.ylabel("Pitch Y")

        return fig
    
    def save_frame(self, path: str):
        fig = self.plot_frame()
        fig.savefig(path)

    def show_frame(self):
        plt.show()

if __name__ == "__main__":

    from xgboost_model import XGBoostModel
    import json

    model = XGBoostModel(model_path="stats/xReceiver/models/xgb_model_no_pc.pkl")

    

    match_id = 666385
    model.load_data(match_id=match_id)

    # Load runs

    runs_path = f"outputs/{match_id}/runs_with_metrics.jsonl"

    runs = []

    with open(runs_path, 'r') as f:
        for line in f:
            runs.append(json.loads(line))


    # get a run

    run = runs[0]

    # Get features

    model.xgb_run_processor.load_run(run)


   
    features = model.xgb_run_processor.get_features(type='start', plyr_id=1159675)

    print(features)

    xP_pred, class_pred  = model.predict(features)

    print(f"xP_pred: {xP_pred}")
    print(f"class_pred: {class_pred}")


    model.visualizer.plot_run()





    


    # visualizer = Visualizer(model)
    # visualizer.set_frame(row_num=row_num)

    

    # fig = visualizer.plot_frame(predictions)
    # visualizer.show_frame()
    
    
