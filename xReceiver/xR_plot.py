
from xR_get_features_new import get_features as get_features

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from xgboost_model import XGBoostModel

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class xR_Plotter:

    def __init__(self, model: 'XGBoostModel'):
        self.model = model

    def get_row_data(self, row_num: int):

        return self.model.merged_jsonl[row_num]
    

    def get_opp_team(self, team_id):

        for _, tm in self.model.wy_id2team_id.items():
            if tm != team_id:
                return tm
    
    def draw_pitch(self, ax):
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.set_facecolor("white")
        ax.add_patch(patches.Rectangle((0, 0), 105, 68, linewidth=2, edgecolor='black', facecolor='#228B22'))  # Grass
        ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], color='white', linewidth=2)  # Outline
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    def plot_pass(self, row_num: int):

        # Get the row data 
        row_data = self.get_row_data(row_num=row_num)

        match_id = self.model.match_info.get('match_id')
        

        event_id = row_data.get('id')
        row_num = row_data.get('row_num')

        # GEt the giver id
        giver_wy_id = row_data.get('player',{}).get('id')
        giver_id = int(self.model.wy_id2plyr_id.get(giver_wy_id))

        # For every pass get the players from same team
        receiver_wy_tm = row_data.get('team',{}).get('id')

        receiver_team_id = int(self.model.wy_id2team_id.get(receiver_wy_tm))
        opp_tm_id = self.get_opp_team(str(receiver_team_id))

        tm_plyrs_pos = row_data.get('data').get(str(receiver_team_id))

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        self.draw_pitch(ax)

        # Draw attacking team

        for plyr in tm_plyrs_pos:
            x, y = plyr.get('x'), 68 - plyr.get('y')
            plyr_id = plyr.get('id')

            # get number 
            jersery_num = self.model.plyr_lookup_dict.get(str(plyr_id), {}).get('number')
            


            if plyr_id == giver_id:

                ax.plot(x, y, '*', color='blue', markersize=35, label='Ball Carrier')
                ax.text(x, y + 0.2 , str(jersery_num), fontsize=15, ha='center', color='white')
                continue

            ax.plot(x, y, 'o', color='blue', markersize=30)
            ax.text(x, y + 0.2 , str(jersery_num), fontsize=15, ha='center', color='white')  

            # Get the xR features for the player
            features = get_features(plyr_id, row_data, self.model.wy_id2team_id, self.model.wy_id2plyr_id, self.model.merged_jsonl, self.model.plyr_lookup_dict, self.model.match_info)

            # Make the xR prediction
            xR_pred, _ = self.model.predict(features)

            # Draw the xR prediction
            ax.text(x, y + 3 , f"xR: {xR_pred:.2f}", fontsize=15, ha='center', color='white')

            

        # Draw defending team
        opp_plyrs_pos = row_data.get('data').get(str(opp_tm_id))
        for plyr in opp_plyrs_pos:
            x, y = plyr.get('x'), 68 - plyr.get('y')
            plyr_id = plyr.get('id')
            jersery_num = self.model.plyr_lookup_dict.get(str(plyr_id), {}).get('number')
            ax.plot(x, y, 'o', color='red', markersize=30)
            ax.text(x, y + 0.2 , str(jersery_num), fontsize=15, ha='center', color='white')

        ax.set_title(f"Pass Event #{row_num} | Match {match_id}", fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.show()

        





if __name__ == "__main__":
    from xgboost_model import XGBoostModel

    model = XGBoostModel(model_path="stats/xReceiver/models/xgb_model_no_pc.pkl")

    

    match_id = 666385
    model.load_data(match_id=match_id)

    row_num = 28457
    row_num = 12927
    

    plotter = xR_Plotter(model)

    plotter.plot_pass(row_num=row_num)
    
    # features = get_features(match_id, row_num, plyr_id)

    # print(features)

    exit()