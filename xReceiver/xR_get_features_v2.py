
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from xgboost_model import XGBoostModel


class xR_Run_Processor:

    def __init__(self, xgb_model: 'XGBoostModel'):
        self.xgb = xgb_model

        


    def load_run(self, run):
        self.run = run

    def get_position_plyr(self, row_data, plyr_id):
        data = row_data.get('data')

        for _, plyrs in data.items():
            for plyr in plyrs:
                if plyr.get('id') == plyr_id:
                    return round(plyr.get('x'),2), round(plyr.get('y'),2)
        return None, None

    def get_dir_vector(self, plyr_coord, plyr_id):

        row_num = self.run_num

        plyr_coord = plyr_coord['x'], plyr_coord['y']

        if row_num == 0:
            return None

        prev_row = self.xgb.merged_jsonl[row_num - 1]

        prev_plyr_coord = self.get_position_plyr(prev_row, plyr_id)

        if prev_plyr_coord[0] == None or plyr_coord[0] == None:   
            return None
        
        return np.array(plyr_coord) - np.array(prev_plyr_coord)  

    @staticmethod
    def calculate_distance(plyr_1_coord, plyr_2_coord):

        giver_pos = plyr_1_coord['x'], plyr_1_coord['y']
        receiver_pos = plyr_2_coord['x'], plyr_2_coord['y']

        
        return np.linalg.norm(np.array(giver_pos) - np.array(receiver_pos))  
    

    # Function to check if the team id is home or away
    def is_home(self, team_id:int) -> bool:

        if team_id == self.xgb.match_info['home_team_wy_id'] or team_id == self.xgb.match_info['home_team_id']:
            return True
        elif team_id == self.xgb.match_info['away_team_wy_id'] or team_id == self.xgb.match_info['away_team_id']:
            return False
        else:
            return None
        
    # Function to get the attacking side
    def get_attacking_side(self, team_id, half):

        is_home_team = self.is_home(team_id)
        

        if is_home_team == None:
            return None

        attacking_side = None
        side_attack = self.xgb.match_info['side_attack']
        
        if side_attack == True:
            if half == 1:
                attacking_side = "left"
            else:
                attacking_side = "right"
        else:
            if half == 1:
                attacking_side = "right"
            else:
                attacking_side = "left"

        # Flip if the match_side is away
        if is_home_team == False:
            if attacking_side == "left":
                attacking_side = "right"
            else:
                attacking_side = "left"

        
        return attacking_side
    
    def get_attack_goal(self, team_id, half):
        attacking_side = self.get_attacking_side(team_id, half)
        if attacking_side == "left":
            return (0, 34)
        else:
            return (105 , 34)
        
    @staticmethod
    def safe_normalize(vector):
        norm = np.linalg.norm(vector)
        if norm == 0 or np.isclose(norm, 0):
            return np.zeros_like(vector)
        return vector / norm
    # Function to calculate the angle between facing direction and line of sight between two players
    def calculate_angle(self, P1, V1, P2, V2):

        P1 = P1['x'], P1['y']
        P2 = P2['x'], P2['y']

        if V1 is None or V2 is None:
            return None , None
        # Step 2: Calculate the direction vectors D1 and D2
        D1 = np.array(P2) - np.array(P1)  # Vector from Player 1 to Player 2
        D2 = np.array(P1) - np.array(P2)  # Vector from Player 2 to Player 1

        # Normalize the facing direction vectors and direction vectors
        # V1_norm = V1 / np.linalg.norm(V1)
        # D1_norm = D1 / np.linalg.norm(D1)
        # V2_norm = V2 / np.linalg.norm(V2)
        # D2_norm = D2 / np.linalg.norm(D2)

        # Normalize the facing direction vectors and direction vectors
        V1_norm = self.safe_normalize(V1)
        D1_norm = self.safe_normalize(D1)
        V2_norm = self.safe_normalize(V2)
        D2_norm = self.safe_normalize(D2)

        # Step 3: Calculate the angle (theta1) between V1 and D1
        cos_theta1 = np.dot(V1_norm, D1_norm)
        cos_theta1 = np.clip(cos_theta1, -1.0, 1.0)  # Clip to avoid precision errors
        theta1 = np.degrees(np.arccos(cos_theta1))

        # Step 3: Calculate the angle (theta2) between V2 and D2
        cos_theta2 = np.dot(V2_norm, D2_norm)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)  # Clip to avoid precision errors
        theta2 = np.degrees(np.arccos(cos_theta2))

        return theta1, theta2
    
    def calc_dir_angle(self, P1, P2):

        P1 = P1['x'], P1['y']
        P2 = P2['x'], P2['y']

        # P1 and P2 are (x, y) coordinates for the pass start and end points
        D_pass = np.array(P2) - np.array(P1)  # Direction vector of the pass

        if np.linalg.norm(D_pass) == 0:
            return 0

        # Define the playing direction vector towards the opponent's goal
        if self.attack_goal[0] == 0:
            D_goal = np.array([-1, 0])
        else:
            D_goal = np.array([1, 0])
        
        # Calculate the dot product
        dot_product = np.dot(D_pass, D_goal)
        
        # Calculate the magnitude of the pass direction vector
        magnitude_pass = np.linalg.norm(D_pass)
        
        # Compute the cosine of the angle
        cos_theta = dot_product / magnitude_pass
        
        # Calculate the angle in radians and convert to degrees
        theta_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        theta_degrees = np.degrees(theta_radians)
        
        return theta_degrees
    

    def get_speed(self, plyr_coord, plyr_id):
    
        row_num = self.run_num
        row_time = self.xgb.merged_jsonl[row_num].get('Videotimestamp')
        if row_num == 0:
            return None

        prev_row = self.xgb.merged_jsonl[row_num - 1]
        prev_time = prev_row.get('Videotimestamp')

        prev_plyr_coord = self.get_position_plyr(prev_row, plyr_id)


        prev_plyr_coord = {
            'x': prev_plyr_coord[0],
            'y': prev_plyr_coord[1]
        }



        
        # Calculate speed
        distance = self.calculate_distance(prev_plyr_coord, plyr_coord)
        time_diff = row_time - prev_time
        speed = distance / time_diff
        return speed
    
    def count_plyr_radius(self, plyr_id, plyr_coord, radius, same_team:bool):

        plyr_id_list = self.run['off_ball_teammates'] if same_team else self.run['defending_team_players']

        count = 0

        for plyr in plyr_id_list:

            if plyr.get('player_id') == plyr_id:
                continue

            target_plyr_coord = plyr.get(f'{self.type}_position')

            if target_plyr_coord == None:
                print(f"Plyr Coord is None for {plyr.get('player_id')}")
                continue
            
            dist = self.calculate_distance(plyr_coord, target_plyr_coord)

            if dist <= radius:
                count += 1

        return count
    

    def num_opp2plyr_vs_goal(self, plyr_pos):
        """
        Counts the number of opponents closer to the goal than the player
        Returns the number of opponents and the closest opponent distance
        """

        # Number of Opponents
        count = 0

        # Closest Opponent Distance
        closest_opp_dist = 999999   

        # Last Defender Info
        last_defender_dist = 999999
        last_defender_x = None

        opp_plyrs = self.run['defending_team_players']

        for opp_plyr in opp_plyrs:

            opp_plyr_pos = opp_plyr.get(f'{self.type}_position')


            attack_goal = {'x': self.attack_goal[0], 'y': self.attack_goal[1]}

            dist2goal = self.calculate_distance(opp_plyr_pos, attack_goal)

            dist2plyr = self.calculate_distance(plyr_pos, opp_plyr_pos)

            # Update the closest opponent distance
            if dist2plyr < closest_opp_dist:
                closest_opp_dist = dist2plyr

            # If the distance to the goal is less than the distance to the opponent, increment the count
            if dist2goal < dist2plyr:
                count += 1

            # Get the last defender

            if dist2goal < last_defender_dist:
                last_defender_dist = dist2goal
                last_defender_x = opp_plyr_pos['x']
    

        return count, closest_opp_dist, last_defender_x
    

    # Function to check if a defender is in the corridor between two points
    @staticmethod
    def is_defender_in_corridor(pt1, pt2, defender, corridor_width=10):

        if pt1['x']:
            pt1 = pt1['x'], pt1['y']

        if pt2['x']:
            pt2 = pt2['x'], pt2['y']

        if defender['x']:
            defender = defender['x'], defender['y']

        # Convert points to numpy arrays (2D)
        pt1 = np.array(pt1)
        pt2 = np.array(pt2)
        defender = np.array(defender)
        
        # Vector from pt1 to pt2 (the pass line)
        pass_vector = pt2 - pt1
        pass_reverse_vector = pt1 - pt2
        
        # Vector from pt1 to the defender
        defender_vector1 = defender - pt1

        # Vector from pt2 to the defender
        defender_vector2 = defender - pt2
        
        # Calculate the length of the pass (magnitude of the pass vector)
        pass_length = np.linalg.norm(pass_vector)
        
        # If the pass length is zero (which shouldn't happen), return False
        if pass_length == 0:
            return False
        
        # Manually calculate the 2D cross product and distance
        cross_product_2d = (pass_vector[0] * defender_vector1[1]) - (pass_vector[1] * defender_vector1[0])
        distance = np.abs(cross_product_2d) / pass_length

        # Calculate the dot product between the pass vector and the defender vector
        # From both the perspective of the start and end of the pass

        dot_product1 = np.dot(pass_vector, defender_vector1)
        dot_product2 = np.dot(pass_reverse_vector, defender_vector2)
        
        # Check if the defender is within the corridor width (corridor_width/2 on either side)
        in_corridor_width =  distance <= (corridor_width / 2)

        if in_corridor_width and dot_product1 > 0 and dot_product2 > 0:
            return True
        else:
            return False
        
    def get_opp_in_corridor(self):

        opp_count = 0

        for opp_plyr in self.run['defending_team_players']:
            

            # Get defender position
            opp_plyr_coord = opp_plyr.get(f'{self.type}_position')


            in_corridor = self.is_defender_in_corridor(self.giver_coord, self.receiver_coord, opp_plyr_coord)

            if in_corridor:
                opp_count += 1

        return opp_count
    

    def get_dist_offside(self):

        receiver_x = self.receiver_coord['x']
        if self.attack_goal[0] == 0:
            dist2offside = receiver_x - self.last_defender_x
        else:
            dist2offside = self.last_defender_x - receiver_x

        if dist2offside >= 0:
            offside = False
        else:
            offside = True

        return dist2offside, offside


    def get_output_dict(self):
        
        output_dict = {}

        output_dict['distance'] = self.distance
        output_dict['theta1'] = self.theta1
        output_dict['theta2'] = self.theta2
        output_dict['dir_angle'] = self.dir_angle
        output_dict['giver_coord.x'] = self.giver_coord['x']
        output_dict['giver_coord.y'] = self.giver_coord['y']
        output_dict['receiver_coord.x'] = self.receiver_coord['x']
        output_dict['receiver_coord.y'] = self.receiver_coord['y']
        output_dict['giver_speed'] = self.giver_speed
        output_dict['receiver_speed'] = self.receiver_speed
        output_dict['tm_count_5'] = self.tm_count_5
        output_dict['tm_count_10'] = self.tm_count_10
        output_dict['opp_count_2'] = self.opp_count_2
        output_dict['opp_count_5'] = self.opp_count_5
        output_dict['opp_count_10'] = self.opp_count_10
        output_dict['dist_giv_sideline'] = self.dist_giv_sideline
        output_dict['dist_rec_sideline'] = self.dist_rec_sideline
        output_dict['num_opp2plyr_goal_giv'] = self.num_opp2plyr_goal_giv
        output_dict['num_opp2plyr_goal_rec'] = self.num_opp2plyr_goal_rec
        output_dict['dist_giv_goal'] = self.dist_giv_goal
        output_dict['dist_rec_goal'] = self.dist_rec_goal
        output_dict['dist_opponent_giv'] = self.closest_opp_dist_rec
        output_dict['dist_opponent_rec'] = self.closest_opp_dist_giv
        output_dict['opp_in_corridor_count'] = self.opp_in_corridor_count
        output_dict['dist2offside'] = self.dist_offside
        output_dict['offside'] = self.offside

        return output_dict
        
    

    def get_features(self, type:str, plyr_id:int):

        if type not in ['start', 'end']:
            raise ValueError(f"Invalid type: {type}")
        
        self.type = type

        self.plyr_id = plyr_id


        self.run_num = self.run[f'{type}_frame'] 

        self.team_id = self.run['team_id']


        # Get the Off Ball Teammates
        self.off_ball_teammates = self.run['off_ball_teammates']

        # Get the Attack Goal
        self.attack_goal = self.get_attack_goal(self.team_id, self.run['period'])

        # Get the Giver Coord
        self.giver_id = self.run[f'ball_carrier_id_{self.type}']

        for plyr in self.off_ball_teammates:
            if int(plyr['player_id']) == int(self.giver_id):
                self.giver_coord= plyr[f'{self.type}_position']


        # Get the Giver Dir Vector
        self.giver_dir_vector = self.get_dir_vector(self.giver_coord, self.giver_id)

        # Get the target Players Coord
        self.receiver_coord = None

        for plyr in self.off_ball_teammates:
            if int(plyr['player_id']) == int(self.plyr_id):
                self.receiver_coord = plyr[f'{self.type}_position']

        # If it is not found, it means that the target player is the runner
        if self.receiver_coord == None:
            self.receiver_coord = self.run['runner_info'][f'{self.type}_position']

        # Get the Receiver Dir Vector
        self.receiver_dir = self.get_dir_vector(self.receiver_coord, self.plyr_id)

        # Calculate the Distance Between the Giver and the Receiver
        self.distance = self.calculate_distance(self.giver_coord, self.receiver_coord)

        # Calculate Angles
        self.theta1, self.theta2 = self.calculate_angle(self.giver_coord, self.giver_dir_vector, self.receiver_coord, self.receiver_dir)

        # Calculate Directional Angle
        self.dir_angle = self.calc_dir_angle(self.giver_coord, self.receiver_coord)

        # Calculate Speeds
        self.giver_speed = self.get_speed(self.giver_coord, self.giver_id)
        self.receiver_speed = self.get_speed(self.receiver_coord, self.plyr_id)

        # Number of team mates within 5 and 10 meters of giver
        self.tm_count_5 = self.count_plyr_radius(self.giver_id, self.giver_coord, radius=5, same_team=True)
        self.tm_count_10 = self.count_plyr_radius(self.giver_id, self.giver_coord, radius=10, same_team=True)

        # Number of opposition players within 2,5 and 10 meters of receiver
        self.opp_count_2 = self.count_plyr_radius(self.plyr_id, self.receiver_coord, radius=2, same_team=False)
        self.opp_count_5 = self.count_plyr_radius(self.plyr_id, self.receiver_coord, radius=5, same_team=False)
        self.opp_count_10 = self.count_plyr_radius(self.plyr_id, self.receiver_coord, radius=10, same_team=False)

        # Distance sideline -> (Distance between the location of the pass and the closest sideline.)
        self.dist_giv_sideline = min(self.giver_coord['x'], 100 - self.giver_coord['x'])
        self.dist_rec_sideline = min(self.receiver_coord['x'], 100 - self.receiver_coord['x'])


        # Number of opponents closer to the goal than the receiver or giver
        self.num_opp2plyr_goal_giv, self.closest_opp_dist_giv, self.last_defender_x = self.num_opp2plyr_vs_goal(self.giver_coord)
        self.num_opp2plyr_goal_rec, self.closest_opp_dist_rec, _ = self.num_opp2plyr_vs_goal(self.receiver_coord)
        

        # Distance to goal for giver and receiver
        self.dist_giv_goal = self.calculate_distance(self.giver_coord, {'x': self.attack_goal[0], 'y': self.attack_goal[1]})
        self.dist_rec_goal = self.calculate_distance(self.receiver_coord, {'x': self.attack_goal[0], 'y': self.attack_goal[1]})

        # Number of opponents in the corridor between the giver and the receiver
        self.opp_in_corridor_count = self.get_opp_in_corridor()

        # Distance to offside and if the player is offside
        self.dist_offside, self.offside = self.get_dist_offside()

        # Get the output dict
        self.features = self.get_output_dict()

        return self.features




if __name__ == "__main__":

    from xgboost_model import XGBoostModel

    run = {'match_id': '666385',
 'run_id': '666385_375',
 'team_id': 3651,
 'player_id': 1157794,
 'ball_carrier_id_start': 1610030,
 'ball_carrier_id_end': 1157794,
 'start_frame': 46125,
 'end_frame': 46161,
 'start_timestamp': 4641.74,
 'end_timestamp': 4645.34,
 'period': 2,
 'runner_info': {'start_position': {'x': 62.0, 'y': 54.56},
  'positions': [{'x': 62.0, 'y': 54.56, 'frame': 46125, 'period': 2}],
  'metrics': {'xR_start': None,
   'xR_end': None,
   'PC_start': None,
   'PC_end': None,
   'xG_start': None,
   'xG_end': None,
   'OBSO_start': None,
   'OBSO_end': None},
  'end_position': {'x': 85.08, 'y': 53.14}},
 'possession': {'start_team': '3651', 'end_team': '3651'},
 'off_ball_teammates': [{'player_id': 38689,
   'start_position': {'x': 52.45, 'y': 23.44},
   'end_position': {'x': 83.35, 'y': 28.88},
   'metrics': {'xR_start': None,
    'xR_end': None,
    'PC_start': None,
    'PC_end': None,
    'xG_start': None,
    'xG_end': None,
    'OBSO_start': None,
    'OBSO_end': None}},
  {'player_id': 1219428,
   'start_position': {'x': 34.68, 'y': 39.08},
   'end_position': {'x': 47.31, 'y': 42.83},
   'metrics': {'xR_start': None,
    'xR_end': None,
    'PC_start': None,
    'PC_end': None,
    'xG_start': None,
    'xG_end': None,
    'OBSO_start': None,
    'OBSO_end': None}},
  {'player_id': 1329659,
   'start_position': {'x': 54.71, 'y': 48.95},
   'end_position': {'x': 75.68, 'y': 48.92},
   'metrics': {'xR_start': None,
    'xR_end': None,
    'PC_start': None,
    'PC_end': None,
    'xG_start': None,
    'xG_end': None,
    'OBSO_start': None,
    'OBSO_end': None}},
  {'player_id': 1263688,
   'start_position': {'x': 32.11, 'y': 40.15},
   'end_position': {'x': 45.54, 'y': 40.96},
   'metrics': {'xR_start': None,
    'xR_end': None,
    'PC_start': None,
    'PC_end': None,
    'xG_start': None,
    'xG_end': None,
    'OBSO_start': None,
    'OBSO_end': None}},
  {'player_id': 45067,
   'start_position': {'x': 47.73, 'y': 64.87},
   'end_position': {'x': 56.63, 'y': 63.98},
   'metrics': {'xR_start': None,
    'xR_end': None,
    'PC_start': None,
    'PC_end': None,
    'xG_start': None,
    'xG_end': None,
    'OBSO_start': None,
    'OBSO_end': None}},
  {'player_id': 1610030,
   'start_position': {'x': 39.2, 'y': 43.42},
   'end_position': {'x': 64.04, 'y': 46.12},
   'metrics': {'xR_start': None,
    'xR_end': None,
    'PC_start': None,
    'PC_end': None,
    'xG_start': None,
    'xG_end': None,
    'OBSO_start': None,
    'OBSO_end': None}},
  {'player_id': 69811,
   'start_position': {'x': 32.83, 'y': 40.78},
   'end_position': {'x': 48.04, 'y': 39.83},
   'metrics': {'xR_start': None,
    'xR_end': None,
    'PC_start': None,
    'PC_end': None,
    'xG_start': None,
    'xG_end': None,
    'OBSO_start': None,
    'OBSO_end': None}},
  {'player_id': 239959,
   'start_position': {'x': 36.04, 'y': 33.96},
   'end_position': {'x': 50.87, 'y': 33.48},
   'metrics': {'xR_start': None,
    'xR_end': None,
    'PC_start': None,
    'PC_end': None,
    'xG_start': None,
    'xG_end': None,
    'OBSO_start': None,
    'OBSO_end': None}},
  {'player_id': 1159675,
   'start_position': {'x': 30.96, 'y': 41.56},
   'end_position': {'x': 45.58, 'y': 44.05},
   'metrics': {'xR_start': None,
    'xR_end': None,
    'PC_start': None,
    'PC_end': None,
    'xG_start': None,
    'xG_end': None,
    'OBSO_start': None,
    'OBSO_end': None}}],
 'defending_team_players': [{'player_id': 1302464,
   'start_position': {'x': 31.14, 'y': 32.3},
   'end_position': {'x': 47.85, 'y': 32.43}},
  {'player_id': 1157731,
   'start_position': {'x': 42.14, 'y': 35.56},
   'end_position': {'x': 59.7, 'y': 34.05}},
  {'player_id': 69834,
   'start_position': {'x': 53.22, 'y': 58.44},
   'end_position': {'x': 81.17, 'y': 52.78}},
  {'player_id': 1159149,
   'start_position': {'x': 40.0, 'y': 39.86},
   'end_position': {'x': 59.0, 'y': 42.0}},
  {'player_id': 1209072,
   'start_position': {'x': 70.38, 'y': 42.51},
   'end_position': {'x': 98.1, 'y': 36.82}},
  {'player_id': 1322737,
   'start_position': {'x': 62.72, 'y': 51.64},
   'end_position': {'x': 81.54, 'y': 48.3}},
  {'player_id': 69716,
   'start_position': {'x': 64.03, 'y': 36.58},
   'end_position': {'x': 81.52, 'y': 38.0}},
  {'player_id': 1267742,
   'start_position': {'x': 50.82, 'y': 30.76},
   'end_position': {'x': 82.46, 'y': 32.92}},
  {'player_id': 1312313,
   'start_position': {'x': 45.56, 'y': 41.9},
   'end_position': {'x': 63.91, 'y': 46.24}},
  {'player_id': 1158910,
   'start_position': {'x': 47.14, 'y': 19.3},
   'end_position': {'x': 76.78, 'y': 32.38}}]}

    
    xgb_model = XGBoostModel(model_path="stats/xReceiver/models/xgb_model_no_pc.pkl")
    xgb_model.load_data(match_id=666385)

    xgb_run_processor = xR_Run_Processor(xgb_model)



    xgb_run_processor.load_run(run)
    features = xgb_run_processor.get_features(type='start', plyr_id=1159675)

    print(features)

    xP_pred, class_pred  = xgb_model.predict(features)

    print(f"xP_pred: {xP_pred}")
    print(f"class_pred: {class_pred}")

    


