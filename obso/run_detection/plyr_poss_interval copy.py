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

            print(f"CURRENT HOLDER: {current_holder}")
            print(f"START TIME: {start_time}")

            player = event.get("player", {})
            player_id = player.get("id")

            if not player_id:
                continue
            

            event_id = event.get("id")

            
            event_time = float(event["videoTimestamp"])

            print(f"EVENT TIME: {event_time}")

            # if event_id == 2448447093:
            if event_id == 2448446378:
                print(f"EVENT: {event}")
                exit()


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
    output_path = f"outputs/{match_id}/player_possessions_by_interval_test1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved detailed player possession segments to {output_path}")

    return results


if __name__ == "__main__":

    import pickle
    

    match_id = 644676

    
    main(match_id)
    
