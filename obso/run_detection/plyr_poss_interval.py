import json


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# Track possession spans per player inside each team possession interval
def get_player_possession_segments(events, possession_intervals):
    results = []

    for team_id, start_vt, end_vt, period in possession_intervals:

        
        interval_result = {
            "team_id": team_id,
            "start_vt": start_vt,
            "end_vt": end_vt,
            "period": period,
            "player_possessions": []
        }


        

        current_holder = None
        start_time = None

        for event in events:
            if (
                event.get("matchPeriod") != period or
                not (start_vt <= float(event.get("videoTimestamp")) <= end_vt) or
                event.get("team", {}).get("id") != team_id
            ):
                continue


            # If event is a pass, switch player_id to receipient        
            player_id = event.get("pass", {}).get("recipient", {}).get("id") if event.get("pass") else  event.get("player", {}).get("id")
                

            if not player_id:
                continue

            event_time = float(event["videoTimestamp"])

            if player_id != current_holder:
                # If the player has changed, add the previous possession to the result
                if current_holder is not None and start_time is not None:

                    
                    interval_result["player_possessions"].append({
                        "player_id": current_holder,
                        "from": start_time,
                        "to": event_time
                    })
                current_holder = player_id
                start_time = event_time

        # Final possession in the interval
        if current_holder is not None and start_time is not None:
            interval_result["player_possessions"].append({
                "player_id": current_holder,
                "from": start_time,
                "to": end_vt
            })

        results.append(interval_result)

    return results

def get_possesion_by_plyr(timestamp, player_possession_segments, period:int, wy_id2_plyr_id:dict):


    for interval in player_possession_segments:

        interval_period = int(interval["period"][0])
        if interval['start_vt'] <= timestamp <= interval['end_vt'] and interval_period == period:
            

            player_possessions = interval["player_possessions"]

            for possession in player_possessions:
                if possession['from'] <= timestamp <= possession['to']:
                    

                
                    plyr_id = wy_id2_plyr_id[possession['player_id']]
                    return int(plyr_id)
                    



def main(match_id):
    events = load_json(f"outputs/{match_id}/events.json")
    possession_intervals = load_json(f"outputs/{match_id}/possession_intervals.json")

    # Run and output results
    results = get_player_possession_segments(events, possession_intervals)

    # Save or print results
    output_path = f"outputs/{match_id}/player_possessions_by_interval.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved detailed player possession segments to {output_path}")

    return results


if __name__ == "__main__":

    import pickle
    

    match_id = 644676

    
    player_possession_segments = main(match_id)

    timestamp = 27.0
    wy_id2_plyr_id = pickle.load(open(f"outputs/{match_id}/wy_id2plyr_id.pkl", "rb"))
    plyr_id = get_possesion_by_plyr(timestamp, player_possession_segments, 1, wy_id2_plyr_id)
    print(f"plyr_id {plyr_id}")
    
