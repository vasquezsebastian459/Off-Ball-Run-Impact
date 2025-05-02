import subprocess
import multiprocessing
from filelock import FileLock
from pathlib import Path
import pandas as pd
from stats.utils.query_db import QueryDB

# Files used to track processing status
PROCESSED_TXT = "stats/pitch_control/processed_matches.txt"
LOCK_FILE = "stats/pitch_control/processed_matches.lock"

def read_processed_matches():
    if not Path(PROCESSED_TXT).exists():
        return set()
    with open(PROCESSED_TXT, "r") as f:
        return set(int(line.strip()) for line in f)

def add_processed_match(match_id):
    with FileLock(LOCK_FILE):
        with open(PROCESSED_TXT, "a") as f:
            f.write(f"{match_id}\n")

def run_match_preprocessing(match_id):
    try:
        print(f"▶️ Processing match {match_id}")
        subprocess.run(["python", "-m", "stats.match_analyzer", str(match_id)], check=True)

        # python /Users/svasquez/AWS_PC/upload_matches.py --match_id 666385

        subprocess.run(["python", "/Users/svasquez/AWS_PC/upload_matches.py", "--match_id", str(match_id)], check=True)
        add_processed_match(match_id)
        print(f"✅ Done: {match_id}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Subprocess failed for match {match_id}: {e}")
    except Exception as e:
        print(f"⚠️ General error on match {match_id}: {e}")

def main(match_ids):
    processed = read_processed_matches()
    unprocessed = [match_id for match_id in match_ids if match_id not in processed]

    print(f"🧮 {len(unprocessed)} matches left to process")

    num_workers = min(multiprocessing.cpu_count(), 2)

    print(f"NUM WORKERS: {num_workers}")
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_match_preprocessing, unprocessed)

if __name__ == "__main__":

    db = QueryDB()

    query = """
    select dmt.match_id from det_match_tracking dmt 
join t_match tm on dmt.match_id =  tm.match_id 
join t_season ts on tm.season_id = ts.season_id 
where tm.season_id in (817532, 817533, 817534, 817688) 
and tracking_status_id = 10
order by ts_updated desc 
limit 200;
    """

    result = db.run_query(query)

    potential_match_ids = result[1]

    potential_match_ids = [match_id[0] for match_id in potential_match_ids]

    print(len(potential_match_ids))
    print(potential_match_ids)

    print("--------------------------------")



    from stats.pitch_control.files_checker import main as files_checker

    file_name = "merged_tracking_events.jsonl"
    match_ids = files_checker(file_name=file_name)

    processed_match_ids = ['623424', '622090', '622092', '622093', '622094', '622096', '622098', '622100', '622101', '622102', '622104', '622106', '622107', '622109', '622112', '622117', '622118', '622121', '622122', '622123', '622124', '622125', '622126', '622128', '622129', '622130', '622133', '622135', '622136', '622139', '622142', '622144', '622146', '622148', '622149', '622150', '622151', '622152', '622153', '622154', '622155', '622156', '622157', '622158', '622159', '622160', '622161', '622162', '622163', '622164', '622165', '622166', '622177', '622178', '622180', '622181', '622182', '622183', '622184', '622185', '622186', '622187', '622188', '622189', '622190', '622191', '622192', '622193', '622194', '622195', '622196', '622197', '622198', '622199', '622202', '622203', '622204', '622205', '622207', '622208', '622209', '622210', '622211', '622212', '622214', '622216', '622217', '622218', '622220', '622221', '622222', '622223', '622224', '622225', '622226', '622227', '622228', '622229', '622230', '622231', '622233', '622234', '622235', '622236', '622237', '622238', '622239', '622240', '622241', '622242', '622243', '622244', '622245', '622246', '622247', '622248', '622249', '622250', '622251', '622252', '622253', '622254', '622255', '622256', '622257', '622258', '622259', '622260', '622261', '622263', '622264', '622265', '622266', '622267', '622268', '622269', '622270', '622271', '622273', '622274', '622275', '622276', '622278', '622279', '622280', '622281', '622282', '622283', '622284', '622285', '622287', '622288', '622289', '622290', '622291', '622292', '622293', '622294', '622295', '622296', '622297', '622298', '622299', '622300', '622301', '622302', '622303', '622304', '622305', '622306', '622307', '622309', '622310', '622311', '622312', '622314', '622315', '622316', '622317', '622318', '622319', '622320', '622321', '622322', '622323', '622324', '622325', '622326', '622327', '622328', '622329', '622330', '622331', '622332', '622333', '622334', '622335', '622336', '622338', '622339', '622340', '622341', '622342', '622343', '622344', '622345', '622346', '622347', '622348', '622349', '622350', '622351', '622352', '622353', '622354', '622356', '622357', '622359', '622360', '622361', '622363', '622364', '622555', '622593', '622600', '622671', '622687', '622690', '622695', '622708', '622709', '622710', '622711', '622715', '622716', '622717', '622718', '622719', '622721', '622722', '622723', '622724', '622725', '622726', '622727', '622728', '622730', '622731', '622733', '622734', '622735', '622736', '622737', '622738', '622739', '622740', '622741', '622742', '622743', '622744', '622745', '622746', '622747', '622748', '622749', '622750', '622751', '622752', '622753', '622754', '622755', '622756', '622849', '622863', '622864', '622920', '622979', '622984', '622991', '623000', '623005', '623006', '623027', '623033', '623047', '623054', '623055', '623057', '623058', '623059', '623060', '623061', '623063', '623064', '623065', '623066', '623069', '623070', '623071', '623073', '623078', '623080', '623089', '623201', '623249', '623250', '623267', '623272', '623273', '623282', '623284', '623301', '623314', '623319', '623321', '623328', '623332', '623337', '623351', '623352', '623355', '623356', '623357', '623369', '623377', '623401', '623414', '623416', '623422', '623425', '623426', '623427', '623430', '623433', '623435', '623436', '623437', '623438', '623439', '623440', '623441', '623442', '628177', '628213', '628324', '628326', '628335', '638259', '638260', '638261', '638262', '638263', '638264', '638265', '638266', '638267', '638268', '638295', '666385']

    processed_match_ids = [int(match_id) for match_id in processed_match_ids]

    to_process_matches = [match_id for match_id in potential_match_ids if match_id not in processed_match_ids]

    
    print(len(to_process_matches))


    
    main(to_process_matches)

    # python -m stats.pitch_control.run_matches_parallel 