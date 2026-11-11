import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
from ori_config import OUTPUTS_DIR, RAW_DATA_DIR  # noqa: E402
import json
import numpy as np
import pickle
import traceback
import pickle
import pandas as pd


print(f"IMPORTING xR_get_features2")


def get_position_plyr(sample_row, plyr_id):
    data = sample_row.get('data')

    for _, plyrs in data.items():
        for plyr in plyrs:
            if plyr.get('id') == plyr_id:
                return round(plyr.get('x'),2), round(plyr.get('y'),2)
    return None, None


# Distance from Passer
def calculate_distance(giver_pos, receiver_pos):

    if giver_pos[0] == None  or receiver_pos[0] == None:
        return None

    return np.linalg.norm(np.array(giver_pos) - np.array(receiver_pos))


# Function to calculate the angle between facing direction and line of sight between two players
def calculate_angle(P1, V1, P2, V2):

    if V1 is None or V2 is None:
        return None , None
    # Step 2: Calculate the direction vectors D1 and D2
    D1 = np.array(P2) - np.array(P1)  # Vector from Player 1 to Player 2
    D2 = np.array(P1) - np.array(P2)  # Vector from Player 2 to Player 1

    # Normalize the facing direction vectors and direction vectors
    V1_norm = V1 / np.linalg.norm(V1)
    D1_norm = D1 / np.linalg.norm(D1)
    V2_norm = V2 / np.linalg.norm(V2)
    D2_norm = D2 / np.linalg.norm(D2)

    # Step 3: Calculate the angle (theta1) between V1 and D1
    cos_theta1 = np.dot(V1_norm, D1_norm)
    cos_theta1 = np.clip(cos_theta1, -1.0, 1.0)  # Clip to avoid precision errors
    theta1 = np.degrees(np.arccos(cos_theta1))

    # Step 3: Calculate the angle (theta2) between V2 and D2
    cos_theta2 = np.dot(V2_norm, D2_norm)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)  # Clip to avoid precision errors
    theta2 = np.degrees(np.arccos(cos_theta2))

    return theta1, theta2


def calculate_directional_angle(P1, P2, goal_coord):
    # P1 and P2 are (x, y) coordinates for the pass start and end points
    D_pass = np.array(P2) - np.array(P1)  # Direction vector of the pass

    # Define the playing direction vector towards the opponent's goal
    if goal_coord[0] == 0:
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

def get_speed(row, merged_tracking_jsonl, plyr_coord, plyr_id):
    
    row_num = row.get('row_num')
    row_time = row.get('Videotimestamp')
    if row_num == 0:
        return None

    prev_row = merged_tracking_jsonl[row_num - 1]
    prev_time = prev_row.get('Videotimestamp')

    prev_plyr_coord = get_position_plyr(prev_row, plyr_id)

    if prev_plyr_coord[0] == None or plyr_coord[0] == None:   
        return None
    

    # Calculate speed
    distance = calculate_distance(prev_plyr_coord, plyr_coord)
    time_diff = row_time - prev_time
    speed = distance / time_diff
    return speed


def get_dir_vector(row, merged_tracking_jsonl, plyr_coord, plyr_id):
    row_num = row.get('row_num')

    if row_num == 0:
        return None

    prev_row = merged_tracking_jsonl[row_num - 1]

    prev_plyr_coord = get_position_plyr(prev_row, plyr_id)

    if prev_plyr_coord[0] == None or plyr_coord[0] == None:   
        return None
    
    return np.array(plyr_coord) - np.array(prev_plyr_coord)    


# Get number of players given a tm_id and a radius
def count_plyr_radius(row, plyr_id, plyr_pos, team_id:str, radius):
    data = row.get('data')

    count = 0
    for tm, plyrs in data.items():
        if tm == team_id:
            for plyr in plyrs:
                if plyr.get('id') != plyr_id:
                    plyr_coord = (round(plyr.get('x'),2), round(plyr.get('y'),2))
                    if calculate_distance(plyr_pos, plyr_coord) <= radius:
                        
                        count += 1

    return count


def get_opp_team(team_id, wy_id2team_id):

    for _, tm in wy_id2team_id.items():
        if tm != team_id:
            return tm


def dist_plyr_sideline(plyr_coord):
    return min(plyr_coord[0], 100 - plyr_coord[0])


# Function to check if the team id is home or away
def is_home(team_id:int, match_info:dict) -> bool:
    if team_id == match_info['home_team_wy_id'] or team_id == match_info['home_team_id']:
        return True
    elif team_id == match_info['away_team_wy_id'] or team_id == match_info['away_team_id']:
        return False
    else:
        return None
    
# Function to get the attacking side
def get_attacking_side(team_id, half, match_info):

    is_home_team = is_home(team_id, match_info)
    

    if is_home_team == None:
        return None

    attacking_side = None
    side_attack = match_info['side_attack']
    
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

def get_attack_goal(team_id, half, match_info):
    attacking_side = get_attacking_side(team_id, half, match_info)
    if attacking_side == "left":
        return (0, 34)
    else:
        return (105 , 34)
    

def dist_plyr_goal(plyr_coord, goal_coord):
    return calculate_distance(plyr_coord, goal_coord)


def num_opp2plyr_vs_goal(row, plyr_pos, goal_coord, opp_tm):

    data = row.get('data')

    count = 0
    for tm, plyrs in data.items():
        if tm == opp_tm:
            for opp_plyr in plyrs:
                # Calcualte the distance between the opponent player and the goal
                opp_plyr_pos = (round(opp_plyr.get('x'),2), round(opp_plyr.get('y'),2))
                
                
                dist2goal = dist_plyr_goal(opp_plyr_pos, goal_coord)
                

                # Calculate the distance between the opponent player and the player
                dist2plyr = calculate_distance(plyr_pos, (round(opp_plyr.get('x'),2), round(opp_plyr.get('y'),2)))
                
                if dist2goal < dist2plyr:
                    
                    count += 1
    return count


def dist_opponent(row, plyr_pos, opp_tm:str):
    data = row.get('data')

    dist = 999999
    
    for tm, plyrs in data.items():
        if tm == opp_tm:
            for plyr in plyrs:
                opp_plyr_coord = (plyr.get('x'), plyr.get('y'))
                opp_dist = calculate_distance(plyr_pos, opp_plyr_coord)

                if opp_dist < dist:
                    dist = opp_dist
    return dist

def is_defender_in_corridor(pt1, pt2, defender, corridor_width=10):
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
    

def get_opp_in_corridor(row, giver_coord, receiver_coord, opp_tm:str, corridor_width=10):

    data = row.get('data')
    opp_count = 0
    for tm, plyrs in data.items():
        if tm == opp_tm:
            for plyr in plyrs:
                opp_plyr_coord = (plyr.get('x'), plyr.get('y'))
                in_corridor = is_defender_in_corridor(giver_coord, receiver_coord, opp_plyr_coord)
                opp_count += 1 if in_corridor else 0
    return opp_count

def dist_offside(row, plyr_lookup_dict, receiver_id, opp_tm_id, goal_coord):

    # Get the last defender of the opp_tm

    data = row.get('data')
    last_defender = {
        'plyr_id' : None,
        'x' : None,
        'dist2goal' : 9999
    }

    for tm, plyrs in data.items():
        if tm == opp_tm_id:
            for plyr in plyrs:
                position = plyr_lookup_dict.get(str(plyr.get('id')), {}).get('position')
                
                dist2goal =  abs(plyr.get('x') -  goal_coord[0])

                if dist2goal < last_defender.get('dist2goal') and position != 'GK':
                    
                    last_defender = {
                        'plyr_id' : plyr.get('id'),
                        'x' : plyr.get('x'),
                        'dist2goal' : dist2goal
                    }
                
    

    last_defender_x = last_defender.get('x')
    dist2offside = None
    for tm, plyrs in data.items():
        if tm != opp_tm_id:
            for plyr in plyrs:

                if plyr.get('id') == receiver_id:
                    receiver_x = plyr.get('x')
                    

                    if goal_coord[0] == 0 :
                        dist2offside =  receiver_x - last_defender_x
                    else:
                        dist2offside = last_defender_x - receiver_x 
                    break
                    
    offside = None
    if not dist2offside:
        return None, None
    if dist2offside >=0:
        offside = False
    else:
        offside = True

    return dist2offside, offside


def get_features(plyr_id, row, wy_id2team_id, wy_id2plyr_id, merged_tracking_jsonl, plyr_lookup_dict, match_info):

    match_id = match_info.get('match_id')
    features = {}

    event_id = row.get('id')
    row_num = row.get('row_num')
    
    # For every pass get the players from same team
    receiver_wy_tm = row.get('team',{}).get('id')

    receiver_team_id = int(wy_id2team_id.get(receiver_wy_tm))
    opp_tm_id = get_opp_team(str(receiver_team_id), wy_id2team_id)

    tm_plyrs_pos = row.get('data').get(str(receiver_team_id))

    giver_wy_id = row.get('player',{}).get('id')

    # Get the real receiver of the pass
    true_receiver_wy_id = row.get('pass').get('recipient').get('id')
    true_receiver_id = wy_id2plyr_id.get(true_receiver_wy_id)
    true_receiver_id = int(true_receiver_id) if true_receiver_id else None

    

    if giver_wy_id == 0 or true_receiver_wy_id == 0:
        return None
    giver_id = int(wy_id2plyr_id.get(giver_wy_id))

    # GEt the position and direction of the Giver Player
    giver_coord = get_position_plyr(row, giver_id)

    if giver_coord[0] == None:
        return None
    giver_dir = get_dir_vector(row, merged_tracking_jsonl, giver_coord, giver_id) 

    # Get the current half
    half = row.get('matchPeriod')
    half = 1 if half == '1H' else 2

    # Get the attacking goal coord
    goal_coord = get_attack_goal(receiver_team_id, half, match_info)

    for plyr in tm_plyrs_pos:
        
        if plyr_id != plyr.get('id'):
            continue

        receiver_id = plyr.get('id')

        # GEt the position and direction of the Giver Player
        receiver_coord = get_position_plyr(row, receiver_id)
        receiver_dir = get_dir_vector(row, merged_tracking_jsonl, receiver_coord, receiver_id)

        # Calculate Features
        distance = calculate_distance(giver_coord, receiver_coord)
        theta1, theta2 = calculate_angle(giver_coord, giver_dir, receiver_coord, receiver_dir)

        # Calculate Directional Angle
        directional_angle = calculate_directional_angle(giver_coord, receiver_coord, goal_coord)

        # Calculate speed of giver and receiver

        giver_speed = get_speed(row, merged_tracking_jsonl, giver_coord, giver_id)
        receiver_speed = get_speed(row, merged_tracking_jsonl, receiver_coord, receiver_id)

        
        # Number of team mates within 5 and 10 meters of giver
        tm_count_5 = count_plyr_radius(row, giver_id, giver_coord, str(receiver_team_id), 5)
        tm_count_10 = count_plyr_radius(row, giver_id, giver_coord, str(receiver_team_id), 10)

        # Number of opposition players within 2,5 and 10 meters of receiver
        opp_count_2 = count_plyr_radius(row, receiver_id, receiver_coord, opp_tm_id, 2)
        opp_count_5 = count_plyr_radius(row, receiver_id, receiver_coord, opp_tm_id, 5)
        opp_count_10 = count_plyr_radius(row, receiver_id, receiver_coord, opp_tm_id, 10)


        # Distance sideline -> (Distance between the location of the pass and the closest sideline.)
        dist_giv_sideline = dist_plyr_sideline(giver_coord)
        dist_rec_sideline = dist_plyr_sideline(receiver_coord)


        # Number of opponents closer to the goal than the receiver or giver
        num_opp2plyr_goal_giv = num_opp2plyr_vs_goal(row, giver_coord, goal_coord, opp_tm_id)
        num_opp2plyr_goal_rec = num_opp2plyr_vs_goal(row, receiver_coord, goal_coord, opp_tm_id)
        

        # Distance to goal for giver and receiver
        dist_giv_goal = dist_plyr_goal(giver_coord, goal_coord)
        dist_rec_goal = dist_plyr_goal(receiver_coord, goal_coord)

        # Distance to closest opponent
        dist_opponent_giv = dist_opponent(row, giver_coord, opp_tm_id)
        dist_opponent_rec = dist_opponent(row, receiver_coord, opp_tm_id)

        # Opponent in the corridor
    
        
        # get_opp_in_corridor(giver_coord, receiver_coord, dist_opponent_rec)
        opp_in_corridor_count = get_opp_in_corridor(row, giver_coord, receiver_coord, opp_tm_id)

        # Get the distance to offisde and if the player is offside
        dist2offside, offised = dist_offside(row, plyr_lookup_dict, receiver_id, opp_tm_id, goal_coord)

        # Label 0 if the receiver is not the true receiver else 1
        target = 1 if receiver_id == true_receiver_id else 0

        features['match_id'] = match_id
        features['event_id'] = event_id
        features['row_num'] = row_num
        features['giver_id'] = giver_id
        features['receiver_id'] = receiver_id
        features['distance'] = round(distance,2) if distance else None
        features['theta1'] = round(theta1,2) if theta1 else None
        features['theta2'] = round(theta2,2) if theta2 else None
        features['dir_angle'] = round(directional_angle,2)
        features['giver_coord.x'] = giver_coord[0]
        features['giver_coord.y'] = giver_coord[1]
        features['receiver_coord.x'] = receiver_coord[0]
        features['receiver_coord.y'] = receiver_coord[1]
        features['giver_speed'] = round(giver_speed,2) if giver_speed else None
        features['receiver_speed'] = round(receiver_speed,2) if receiver_speed else None
        features['tm_count_5'] = tm_count_5
        features['tm_count_10'] = tm_count_10
        features['opp_count_2'] = opp_count_2
        features['opp_count_5'] = opp_count_5
        features['opp_count_10'] = opp_count_10
        features['dist_giv_sideline'] = dist_giv_sideline
        features['dist_rec_sideline'] = dist_rec_sideline
        features['num_opp2plyr_goal_giv'] = num_opp2plyr_goal_giv
        features['num_opp2plyr_goal_rec'] = num_opp2plyr_goal_rec
        features['dist_giv_goal'] = round(dist_giv_goal,2)
        features['dist_rec_goal'] = round(dist_rec_goal,2)
        features['dist_opponent_giv'] = round(dist_opponent_giv,2)
        features['dist_opponent_rec'] = round(dist_opponent_rec,2)
        features['opp_in_corridor_count'] = int(opp_in_corridor_count)
        features['dist2offside'] = round(dist2offside,2) if dist2offside else None
        features['offside'] = 1 if offised else 0
        features['target'] = target

        
        

    return features


def main(match_id, row_num, plyr_id):

    base_dir = f"{OUTPUTS_DIR}/{match_id}"
    game_path = base_dir + "/merged_tracking_events.jsonl"

    with open(game_path) as f:
        merged_tracking_jsonl = list(map(json.loads, f.read().splitlines()))

    wy_id2plyr_id_path = base_dir + "/wy_id2plyr_id.pkl"
    wy_id2team_id_path = base_dir + "/wy_id2team_id.pkl"
    match_info = base_dir + "/cached_data/match_info.json"
    plyr_lookup_path = base_dir + "/plyr_lookup_dict.json"

    # Open the file with pickle
    with open(wy_id2plyr_id_path, "rb") as f:
        wy_id2plyr_id = pickle.load(f)


    # Open the file with pickle
    with open(wy_id2team_id_path, "rb") as f:
        wy_id2team_id = pickle.load(f)

    # Open the file with json
    with open(match_info, "r") as f:
        match_info = json.load(f)

    # Open Player Lookup Dictionary
    with open(plyr_lookup_path, "r") as f:
        plyr_lookup_dict = json.load(f)

    
    # Take a row num and a playerid
   

    row = merged_tracking_jsonl[row_num]

    # Get the features
    features = get_features(plyr_id, row, wy_id2team_id, wy_id2plyr_id, merged_tracking_jsonl, plyr_lookup_dict, match_info)

    print(features)

    return features


if __name__ == "__main__":
    base_dir = f"{RAW_DATA_DIR}/FRA I 2024/624102"

    match_id = 666385

    row_num = 20
    plyr_id = 151904
    
    main(match_id, row_num, plyr_id)