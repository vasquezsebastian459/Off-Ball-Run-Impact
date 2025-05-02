from math import atan2, degrees
import numpy as np
import joblib
import json
import pandas as pd
class xGModel:

    def __init__(self, match_id, model_path, feature_names_path= None):
        self.match_id = match_id
        self.model = joblib.load(model_path)

        if feature_names_path is not None:

            self.features_cols = joblib.load(feature_names_path)

        else:
            self.features_cols = self.model.get_booster().feature_names


        self.load_match_data()

    

    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        return data

    def load_match_data(self):
        self.match_info = self.load_json(f"outputs/{self.match_id}/cached_data/match_info.json")

        # Load the merged jsonl

        merged_path = f"outputs/{self.match_id}/merged_tracking_events.jsonl"
        self.merged_jsonl = []
        with open(merged_path, 'r') as f:
            for line in f:
                self.merged_jsonl.append(json.loads(line))
    

    def load_run(self, run):
        self.run = run

        # Function to check if the team id is home or away
    def is_home(self, team_id:int) -> bool:

        if team_id == self.match_info['home_team_wy_id'] or team_id == self.match_info['home_team_id']:
            return True
        elif team_id == self.match_info['away_team_wy_id'] or team_id == self.match_info['away_team_id']:
            return False
        else:
            return None
        
    # Function to get the attacking side
    def get_attacking_side(self, team_id, half):

        is_home_team = self.is_home(team_id)
        

        if is_home_team == None:
            return None

        attacking_side = None
        side_attack = self.match_info['side_attack']
        
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

        prev_row = self.merged_jsonl[row_num - 1]

        prev_plyr_coord = self.get_position_plyr(prev_row, plyr_id)

        if prev_plyr_coord[0] == None or plyr_coord[0] == None:   
            return None
        
        return np.array(plyr_coord) - np.array(prev_plyr_coord)  
    
    @staticmethod
    def calculate_distance(plyr_1_coord, plyr_2_coord):

        if isinstance(plyr_1_coord, dict):
            plyr_1_coord = plyr_1_coord['x'], plyr_1_coord['y']

        if isinstance(plyr_2_coord, dict):
            plyr_2_coord = plyr_2_coord['x'], plyr_2_coord['y']

        
        return np.linalg.norm(np.array(plyr_1_coord) - np.array(plyr_2_coord))  
    
    @staticmethod
    def safe_normalize(vector):
        norm = np.linalg.norm(vector)
        if norm == 0 or np.isclose(norm, 0):
            return np.zeros_like(vector)
        return vector / norm

    # Function to calculate the angle between facing direction and line of sight between two players
    def calculate_angle(self, P1, V1, P2, V2):

        if isinstance(P1, dict):
            P1 = P1['x'], P1['y']

        if isinstance(P2, dict):
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

        if isinstance(P1, dict):    
            P1 = P1['x'], P1['y']

        if isinstance(P2, dict):
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
    
    @staticmethod
    def calculate_goal_opening_angle(distance_to_goal, goal_width=7.32):
        """
        Calculates the angle (in degrees) of the opening between the goalposts from a given distance.
        """
        if distance_to_goal == 0:
            return 180.0  # practically standing on the goal line
        angle_rad = atan2(goal_width / 2, distance_to_goal)
        return round(degrees(angle_rad * 2), 2)  # multiply by 2 to get full opening angle
    

    def get_speed(self, plyr_coord, plyr_id):
    
        row_num = self.run_num
        row_time = self.merged_jsonl[row_num].get('Videotimestamp')
        if row_num == 0:
            return None

        prev_row = self.merged_jsonl[row_num - 1]
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

        if isinstance(pt1, dict):
            pt1 = pt1['x'], pt1['y']

        if isinstance(pt2, dict):
            pt2 = pt2['x'], pt2['y']

        if isinstance(defender, dict):
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

            in_corridor = self.is_defender_in_corridor(self.shooter_coord, self.attack_goal, opp_plyr_coord)

            if in_corridor:
                opp_count += 1

        return opp_count
    

    def get_dist_offside(self):

        shooter_x = self.shooter_coord['x']
        if self.attack_goal[0] == 0:
            dist2offside = shooter_x - self.last_defender_x
        else:
            dist2offside = self.last_defender_x - shooter_x

        if dist2offside >= 0:
            offside = False
        else:
            offside = True

        return dist2offside, offside


    def get_output_dict(self):
    
        output_dict = {
            'shooter_dir_x': self.dir_vector[0],
            'shooter_dir_y': self.dir_vector[1],
            'distance': self.distance_to_goal,
            'shot_time': self.time,
            'shooter_coord.x': self.shooter_coord['x'],
            'shooter_coord.y': self.shooter_coord['y'],
            'goal_coord.x': self.attack_goal[0],
            'goal_coord.y': self.attack_goal[1],
            'theta1': self.theta1,
            'theta2': self.theta2,
            'goal_opening_angle': self.goal_opening_angle,
            'dir_angle': self.dir_angle,
            'shooter_speed': self.speed,
            'tm_count_5': self.tm_count_5,
            'tm_count_10': self.tm_count_10,
            'opp_count_2': self.opp_count_2,
            'opp_count_5': self.opp_count_5,
            'opp_count_10': self.opp_count_10,
            'dist_shooter_sideline': self.dist_shooter_sideline,
            'num_opp2plyr_goal_shooter': self.num_opp2plyr_goal_shooter,
            'dist_shooter_goal': self.dist_shooter_goal,
            'dist_opponent_shooter': self.closest_opp_dist,
            'opp_in_corridor_count': self.opp_in_corridor,
            'dist2offside': self.dist_offside,
            'offside': self.offside
        }

        return output_dict
    

    def get_features(self, type:str, plyr_id:int, speed_imputation:bool = False):

        if type not in ['start', 'end']:
            raise ValueError("Type must be either 'start' or 'end'")
        
        self.type = type

        self.run_num = self.run[f'{type}_frame'] 

        # Get time
        self.time = self.run.get(f'{type}_timestamp')

        # Get the Off Ball Teammates
        self.off_ball_teammates = self.run['off_ball_teammates']

        # Get the Team ID
        self.team_id = self.run['team_id']

        # Get the Attack Goal
        self.attack_goal = self.get_attack_goal(self.team_id, self.run['period'])

        # Get Shooter ID
        self.shooter_id = plyr_id

        # Get the Shooter's Coord
        self.shooter_coord = None

        for plyr in self.off_ball_teammates:
            if int(plyr['player_id']) == int(self.shooter_id):
                self.shooter_coord = plyr[f'{self.type}_position']

        # If it is not found, it means that the target player is the runner
        if self.shooter_coord == None:
            self.shooter_coord = self.run['runner_info'][f'{self.type}_position']

        # Get the Direction Vector
        self.dir_vector = self.get_dir_vector(self.shooter_coord, self.shooter_id)


        # Get the Distance to the Goal
        self.distance_to_goal = self.calculate_distance(self.shooter_coord, self.attack_goal)


        # Calculate angles
        self.theta1, self.theta2 = self.calculate_angle(self.shooter_coord, self.dir_vector, self.attack_goal, self.dir_vector)

        # Calculate the Angle between the Direction Vector and the Goal
        self.dir_angle = self.calc_dir_angle(self.shooter_coord, self.attack_goal)


        # Calculate the Goal Opening Angle
        self.goal_opening_angle = self.calculate_goal_opening_angle(self.distance_to_goal)


        # Calculate the Speed of the Shooter
        self.speed = self.get_speed(self.shooter_coord, self.shooter_id)

        if speed_imputation and self.speed > 8 : # 99th percentile
            self.speed = 2.3 # Training Data Mean


        # Number of team mates within 5 and 10 meters of giver
        self.tm_count_5 = self.count_plyr_radius(self.shooter_id, self.shooter_coord, radius=5, same_team=True)
        self.tm_count_10 = self.count_plyr_radius(self.shooter_id, self.shooter_coord, radius=10, same_team=True)

        # Number of opposition players within 2,5 and 10 meters of receiver
        self.opp_count_2 = self.count_plyr_radius(self.shooter_id, self.shooter_coord, radius=2, same_team=False)
        self.opp_count_5 = self.count_plyr_radius(self.shooter_id, self.shooter_coord, radius=5, same_team=False)
        self.opp_count_10 = self.count_plyr_radius(self.shooter_id, self.shooter_coord, radius=10, same_team=False)

        # Distance sideline -> (Distance between the location of the pass and the closest sideline.)
        self.dist_shooter_sideline = min(self.shooter_coord['x'], 100 - self.shooter_coord['x'])

        # Number of opponents closer to the goal than shooter
        self.num_opp2plyr_goal_shooter,  self.closest_opp_dist, self.last_defender_x = self.num_opp2plyr_vs_goal(self.shooter_coord)

        
        self.dist_shooter_goal = self.calculate_distance(self.shooter_coord, {'x': self.attack_goal[0], 'y': self.attack_goal[1]})


        # Get opp in corridor
        self.opp_in_corridor = self.get_opp_in_corridor()


        # Get dist offside
        self.dist_offside, self.offside = self.get_dist_offside()


        # Get the output dict
        self.features = self.get_output_dict()

        return self.features
    
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

    run = {'match_id': 622240, 'run_id': '622240_114', 'team_id': 4279, 'player_id': 1776309, 'ball_carrier_id_start': 80877, 'ball_carrier_id_end': 1477163, 'start_frame': 13040, 'end_frame': 13088, 'start_timestamp': 1305.3, 'end_timestamp': 1310.1, 'period': 1, 'runner_info': {'start_position': {'x': 49.0, 'y': 4.29}, 'positions': [{'x': 49.0, 'y': 4.29, 'frame': 13040, 'period': 1}], 'metrics': {'xR_start': 0.4180000126361847, 'xR_end': 0.5479999780654907, 'PC_start': 0.549, 'PC_end': 0.565, 'xG_start': 0.084, 'xG_end': 0.084, 'OBSO_start': None, 'OBSO_end': None}, 'end_position': {'x': 29.84, 'y': 25.2}}, 'possession': {'start_team': '4279', 'end_team': '4279'}, 'off_ball_teammates': [{'player_id': 1216616, 'start_position': {'x': 34.06, 'y': 12.52}, 'end_position': {'x': 29.33, 'y': 19.06}, 'metrics': {'xR_start': 0.1889999955892563, 'xR_end': 0.7099999785423279, 'PC_start': 0.728, 'PC_end': 0.842, 'xG_start': 0.057, 'xG_end': 0.101, 'OBSO_start': None, 'OBSO_end': None}}, {'player_id': 1477163, 'start_position': {'x': 42.16, 'y': 5.01}, 'end_position': {'x': 33.26, 'y': 16.14}, 'metrics': {'xR_start': 0.4020000100135803, 'xR_end': 0.6970000267028809, 'PC_start': 0.232, 'PC_end': 0.898, 'xG_start': 0.09, 'xG_end': 0.103, 'OBSO_start': None, 'OBSO_end': None}}, {'player_id': 1221869, 'start_position': {'x': 44.31, 'y': 26.52}, 'end_position': {'x': 42.69, 'y': 33.3}, 'metrics': {'xR_start': 0.20800000429153442, 'xR_end': 0.5659999847412109, 'PC_start': 0.899, 'PC_end': 0.858, 'xG_start': 0.084, 'xG_end': 0.061, 'OBSO_start': None, 'OBSO_end': None}}, {'player_id': 80877, 'start_position': {'x': 60.19, 'y': 3.46}, 'end_position': {'x': 80.43, 'y': 3.12}, 'metrics': {'xR_start': 0.6740000247955322, 'xR_end': 0.6010000109672546, 'PC_start': 0.903, 'PC_end': 1.0, 'xG_start': 0.079, 'xG_end': 0.146, 'OBSO_start': None, 'OBSO_end': None}}, {'player_id': 69936, 'start_position': {'x': 34.26, 'y': 38.28}, 'end_position': {'x': 27.67, 'y': 39.2}, 'metrics': {'xR_start': 0.11599999666213989, 'xR_end': 0.3100000023841858, 'PC_start': 0.638, 'PC_end': 0.643, 'xG_start': 0.06, 'xG_end': 0.073, 'OBSO_start': None, 'OBSO_end': None}}, {'player_id': 1156241, 'start_position': {'x': 60.35, 'y': 52.96}, 'end_position': {'x': 54.93, 'y': 52.9}, 'metrics': {'xR_start': 0.4050000011920929, 'xR_end': 0.37599998712539673, 'PC_start': 1.0, 'PC_end': 1.0, 'xG_start': 0.165, 'xG_end': 0.133, 'OBSO_start': None, 'OBSO_end': None}}, {'player_id': 1339445, 'start_position': {'x': 51.59, 'y': 14.87}, 'end_position': {'x': 47.02, 'y': 26.08}, 'metrics': {'xR_start': 0.4320000112056732, 'xR_end': 0.6359999775886536, 'PC_start': 0.339, 'PC_end': 0.676, 'xG_start': 0.07, 'xG_end': 0.072, 'OBSO_start': None, 'OBSO_end': None}}, {'player_id': 1746425, 'start_position': {'x': 60.9, 'y': 42.84}, 'end_position': {'x': 44.01, 'y': 49.07}, 'metrics': {'xR_start': 0.42800000309944153, 'xR_end': 0.4230000078678131, 'PC_start': 0.904, 'PC_end': 0.906, 'xG_start': 0.111, 'xG_end': 0.102, 'OBSO_start': None, 'OBSO_end': None}}, {'player_id': 1300221, 'start_position': {'x': 48.95, 'y': 3.24}, 'end_position': {'x': 47.32, 'y': 3.26}, 'metrics': {'xR_start': 0.4880000054836273, 'xR_end': 0.335999995470047, 'PC_start': 0.542, 'PC_end': 0.839, 'xG_start': 0.06, 'xG_end': 0.1, 'OBSO_start': None, 'OBSO_end': None}}], 'defending_team_players': [{'player_id': 138304, 'start_position': {'x': 35.56, 'y': 9.49}, 'end_position': {'x': 26.64, 'y': 20.53}}, {'player_id': 80998, 'start_position': {'x': 35.58, 'y': 38.4}, 'end_position': {'x': 23.41, 'y': 41.74}}, {'player_id': 1313767, 'start_position': {'x': 46.12, 'y': 42.22}, 'end_position': {'x': 35.2, 'y': 42.03}}, {'player_id': 1159693, 'start_position': {'x': 34.91, 'y': 28.37}, 'end_position': {'x': 24.09, 'y': 36.14}}, {'player_id': 1157714, 'start_position': {'x': 52.97, 'y': 6.8}, 'end_position': {'x': 46.96, 'y': 11.02}}, {'player_id': 1777620, 'start_position': {'x': 44.11, 'y': 6.51}, 'end_position': {'x': 34.2, 'y': 25.41}}, {'player_id': 1159349, 'start_position': {'x': 48.48, 'y': 3.04}, 'end_position': {'x': 49.06, 'y': 3.27}}, {'player_id': 1701495, 'start_position': {'x': 41.86, 'y': 18.27}, 'end_position': {'x': 36.68, 'y': 34.76}}, {'player_id': 1526425, 'start_position': {'x': 52.0, 'y': 18.31}, 'end_position': {'x': 44.16, 'y': 26.14}}, {'player_id': 81918, 'start_position': {'x': 34.35, 'y': 18.13}, 'end_position': {'x': 25.13, 'y': 28.75}}], 'att_goal': [0, 34], 'dist_to_att_goal_start': 57.303438814786674, 'dist_to_att_goal_end': 31.110538407427153, 'dist_to_att_goal_diff': -26.19290040735952, 'x_axis_dist_to_att_goal_start': 49.0, 'x_axis_dist_to_att_goal_end': 29.84, 'x_axis_dist_to_att_goal_diff': -19.16}


    
    log_reg_model = f"stats/xG/models/xg_logistic_calibrated_model.pkl"

    feature_names_path = f"stats/xG/models/xg_logistic_calibrated_feature_names.pkl"

    xg_model = xGModel(match_id=622240, model_path=log_reg_model, feature_names_path=feature_names_path)

    xg_model.load_run(run)

    

    features = xg_model.get_features(type='end', plyr_id=69936)

    print(f"FEATURES: {features}")

    xP_pred, class_pred = xg_model.predict(features)

    print(xP_pred, class_pred)




