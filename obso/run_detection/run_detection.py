import time
import json
import math
import pickle
import uuid


from run_detection.plyr_poss_interval import main as get_player_possession_segments, get_possesion_by_plyr
# from plyr_poss_interval import main as get_player_possession_segments, get_possesion_by_plyr

MIN_FRAMES = 15  # Minimum duration in frames for a valid run 

def calculate_speed(pos1, pos2, time_diff=0.1):
    dx = pos2['x'] - pos1['x']
    dy = pos2['y'] - pos1['y']
    dist = math.sqrt(dx**2 + dy**2)
    return dist / time_diff  # m/s

def calculate_acceleration(speed1, speed2, time_diff=0.1):
    return (speed2 - speed1) / time_diff  # m/s²

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_possession(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_wy_id2_team_id(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def get_team_in_possession(timestamp, period, possession_intervals, wy_id2_team_id):

    for team_id, start, end, half in possession_intervals:

        # print(f"Team id: {team_id}, start: {start}, end: {end}, half: {half}")
        # if half == ("1H" if period == 1 else "2H") and start <= timestamp <= end:
        if start <= timestamp <= end:
            
            return wy_id2_team_id[team_id]
    return None

def find_ball_carrier_at_time(timestamp, period, timeline, wy_id2_plyr_id):


    for segment in timeline:

        seg_period = int(segment['period'][0])

        if seg_period == period and segment['from'] <= timestamp <= segment['to']:

            wy_plyr_id = segment['player_id']

            plyr_id = int(wy_id2_plyr_id[wy_plyr_id])

            return plyr_id
    return None


def get_player_position_at_frame(tracking_data_by_frame, frame_number, team_id, player_id):
    frame_data = tracking_data_by_frame.get(frame_number)
    if not frame_data:
        return None
    team_players = frame_data['data'].get(str(team_id), [])
    for p in team_players:
        if p['id'] == player_id:
            return {"x": p['x'], "y": p['y']}
    return None

def add_off_ball_teammates(run, tracking_data_by_frame, GK_ids):
    off_ball_teammates = []
    team_players_ids = set()

    start_frame_data = tracking_data_by_frame.get(run['start_frame'], {})
    if start_frame_data:
        team_data = start_frame_data['data'].get(str(run['team_id']), [])
        team_players_ids = {p['id'] for p in team_data}

    excluded_ids = {run['player_id'], run.get('ball_carrier_id')}
    
    # add the GK ids to the excluded ids
    excluded_ids.update(GK_ids)

    for teammate_id in team_players_ids - excluded_ids:
        start_pos = get_player_position_at_frame(tracking_data_by_frame, run['start_frame'], run['team_id'], teammate_id)
        end_pos = get_player_position_at_frame(tracking_data_by_frame, run['end_frame'], run['team_id'], teammate_id)

        if start_pos and end_pos:
            teammate_info = {
                "player_id": teammate_id,
                "start_position": start_pos,
                "end_position": end_pos,
                "metrics": {
                    "xR_start": None,
                    "xR_end": None,
                    "PC_start": None,
                    "PC_end": None,
                    "xG_start": None,
                    "xG_end": None,
                    "OBSO_start": None,
                    "OBSO_end": None
                }
            }
            off_ball_teammates.append(teammate_info)

    run["off_ball_teammates"] = off_ball_teammates

    return run

def get_defending_team_info(run, tracking_data_by_frame, defending_team_id, GK_ids):
    defending_players_info = []
    team_players_ids = set()

    start_frame_data = tracking_data_by_frame.get(run['start_frame'], {})
    if start_frame_data:
        team_data = start_frame_data['data'].get(str(defending_team_id), [])
        team_players_ids = {p['id'] for p in team_data}

    excluded_ids = set(GK_ids)  # Optionally exclude GKs if you want only outfield defenders

    for defender_id in team_players_ids - excluded_ids:
        start_pos = get_player_position_at_frame(tracking_data_by_frame, run['start_frame'], defending_team_id, defender_id)
        end_pos = get_player_position_at_frame(tracking_data_by_frame, run['end_frame'], defending_team_id, defender_id)

        if start_pos and end_pos:
            defender_info = {
                "player_id": defender_id,
                "start_position": start_pos,
                "end_position": end_pos,
            }
            defending_players_info.append(defender_info)

    return defending_players_info


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    

def get_GK_ids(plyr_lookup):
    GK_ids = []
    for plyr_id, info in plyr_lookup.items():
        
        position = info['position']

        if position == 'GK':
            GK_ids.append(int(plyr_id))


    return GK_ids


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)





def detect_runs(match_id):

    jsonl_path = f"outputs/{match_id}/{match_id}_tracking_data.jsonl"
    possession_path = f"outputs/{match_id}/possession_intervals.json"
    wy_id2_team_id_path = f"outputs/{match_id}/wy_id2team_id.pkl"
    wy_id2_plyr_id_path = f"outputs/{match_id}/wy_id2plyr_id.pkl"
    output_path = f"outputs/{match_id}/{match_id}_runs.jsonl"


    plyr_lookup_path = f"outputs/{match_id}/plyr_lookup_dict.json"

    plyr_lookup = load_json(plyr_lookup_path)

    GK_ids:list[int] = get_GK_ids(plyr_lookup)


    player_possession_segments = get_player_possession_segments(match_id)

    data = load_jsonl(jsonl_path)

    tracking_data_by_frame = {}
    for idx, frame in enumerate(data):
        if idx == 0:
            continue
        tracking_data_by_frame[frame['frame']] = frame


    possession_intervals = load_possession(possession_path)

    
    wy_id2_team_id = load_wy_id2_team_id(wy_id2_team_id_path)

    # Load wy_id2plyr_id
    wy_id2_plyr_id = load_pickle(wy_id2_plyr_id_path)

    player_hist = {}
    runs = []
    run_id_counter = 0


    active_runs = {}


    for idx in range(2, len(data)):

        try:

            print(f"\rProcessing frame {idx} of {len(data)}", end='', flush=True)
            frame = data[idx]
            frame_id = frame['frame']

            frame_id = idx
            timestamp = frame['Videotimestamp']
            period = frame['period']
            team_in_possession = get_team_in_possession(timestamp, period, possession_intervals, wy_id2_team_id)

        

            # ball_carrier = find_ball_carrier_at_time(timestamp, period, timeline, wy_id2_plyr_id)
            ball_carrier = get_possesion_by_plyr(timestamp, player_possession_segments, period, wy_id2_plyr_id)

            


            for team_id, players in frame['data'].items():

                if team_in_possession is None:
                    continue
                if int(team_id) != int(team_in_possession):
                    continue

                # Get defending team id

                defending_team_id = None

                for _, db_team_id in wy_id2_team_id.items():

                    if int(team_id) != int(db_team_id):
                        defending_team_id = db_team_id

                if defending_team_id is None:

                    raise ValueError(f"Defending team id not found for team {team_id}")

                


                for p in players:
                    pid = p['id']
                    pos = {'x': p['x'], 'y': p['y'], 'frame': frame_id, 'period': period}

                    if pid not in player_hist:
                        player_hist[pid] = []
                    player_hist[pid].append(pos)

                    if len(player_hist[pid]) < 3:
                        continue

                    p1, p2, p3 = player_hist[pid][-3:]
                    v1 = calculate_speed(p1, p2)
                    v2 = calculate_speed(p2, p3)
                    acc = calculate_acceleration(v1, v2)

                    if v2 >= 5 and acc >= 2.5 and pid not in active_runs:


                        active_runs[pid] = {
                            "match_id": match_id,
                            "run_id": f"run_{str(uuid.uuid4())[:8]}",
                            "team_id": int(team_id),
                            "player_id": pid,
                            "ball_carrier_id_start": ball_carrier,
                            "ball_carrier_id_end": None,
                            "start_frame": p3['frame'],
                            "end_frame": None,
                            "start_timestamp": timestamp, 
                            "end_timestamp": None,
                            "period": period,

                            "runner_info": {
                                "start_position": {"x": p3['x'], "y": p3['y']},
                                "positions": [p3],
                                "metrics": {
                                    "xR_start": None,
                                    "xR_end": None,
                                    "PC_start": None,
                                    "PC_end": None,
                                    "xG_start": None,
                                    "xG_end": None,
                                    "OBSO_start": None,
                                    "OBSO_end": None
                                }
                            },
                            "possession": {
                                "start_team": team_id,
                                "end_team": team_id
                            },
                            "off_ball_teammates": []
                        }
                        

                    elif pid in active_runs:
                        run = active_runs[pid]
                        
                        same_team = int(team_in_possession) == int(run['team_id']) if team_in_possession else False

                        if v2 <= 4 or acc <= -2.5 or not same_team:
                            duration = p3['frame'] - run['start_frame'] + 1

                            if duration >= MIN_FRAMES:
                                run['end_frame'] = p3['frame']

                                run['end_timestamp'] = timestamp  
                                
                                run['runner_info']['end_position'] = {'x': p3['x'], 'y': p3['y']}

                                # run['ball_carrier_id_end'] = find_ball_carrier_at_time(timestamp, period, timeline, wy_id2_plyr_id)
                                run['ball_carrier_id_end'] = get_possesion_by_plyr(timestamp, player_possession_segments, period, wy_id2_plyr_id)
                                
                                run = add_off_ball_teammates(run, tracking_data_by_frame, GK_ids)

                                run["defending_team_players"] = get_defending_team_info(run, tracking_data_by_frame, defending_team_id, GK_ids)

                                run['run_id'] = f"{match_id}_{run_id_counter}"
                                run_id_counter += 1
                                runs.append(run)

                            
                                
                            del active_runs[pid]
                        else:
                            # run['runner_info']['positions'].append(p3)
                            pass

            for run in list(active_runs.values()):
                duration = run['runner_info']['positions'][-1]['frame'] - run['start_frame'] + 1
                if duration >= MIN_FRAMES:
                    
                    run['end_frame'] = run['runner_info']['positions'][-1]['frame']
                    run['end_timestamp'] = timestamp  
                    run['runner_info']['end_position'] = {
                        'x': run['runner_info']['positions'][-1]['x'],
                        'y': run['runner_info']['positions'][-1]['y']
                    }


                    # run['ball_carrier_id_end'] = find_ball_carrier_at_time(timestamp, period, timeline, wy_id2_plyr_id)
                    run['ball_carrier_id_end'] = get_possesion_by_plyr(timestamp, player_possession_segments, period, wy_id2_plyr_id)

                    run = add_off_ball_teammates(run, tracking_data_by_frame, GK_ids)

                    run['run_id'] = f"{match_id}_{run_id_counter}"
                    run_id_counter += 1
                    
                    runs.append(run)


        except Exception as e:
            print(f"Error processing frame {idx} of {len(data)}: {e}")
            continue

    
    # Save runs to jsonl file
    with open(output_path, 'w') as f:
        for run in runs:
            f.write(json.dumps(run) + '\n')


    print(f"✅ Detected {len(runs)} valid runs (≥ {MIN_FRAMES} frames). Saved to {output_path}.")

    return runs

if __name__ == "__main__":
    match_id = 622203
    match_id = 622210
    # match_id = 623370
    # match_id = 623175
    start_time = time.time()
    detect_runs(match_id)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
