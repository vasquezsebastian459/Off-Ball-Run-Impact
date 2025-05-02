import os


def check_output_folders():
    dir = "outputs"
    folders = []
    try:
        # Get all folders in the outputs directory
        folders = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
        return folders
    except FileNotFoundError:
        print(f"Directory '{dir}' not found")
        return []
    except Exception as e:
        print(f"Error checking folders: {str(e)}")
        return []


def check_pc_files(folder):
    dir = f"outputs/{folder}"
    files = []
    try:
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        return files
    except FileNotFoundError:
        print(f"Directory '{dir}' not found")
        return []
    except Exception as e:
        print(f"Error checking files: {str(e)}")
        return []

def main(file_name):

    match_ids = []
    folders = check_output_folders()
    
    for folder in folders:
        files = check_pc_files(folder)

        if file_name in files:
            match_ids.append(folder)
    
    return match_ids

if __name__ == "__main__":

    import pandas as pd

    training_df = pd.read_csv("stats/obso/datasets/training_df.csv")

    training_match_ids = training_df["match_id"].tolist()
    print(f"TRAINING MATCH IDS: {len(training_match_ids)}")

    file_name = "merged_tracking_events.jsonl"
    match_ids = main(file_name=file_name)
    print(f"FILES FOUND WITH {file_name}: {len(match_ids)}")
    print(match_ids)

    print("--------------------------------")



        