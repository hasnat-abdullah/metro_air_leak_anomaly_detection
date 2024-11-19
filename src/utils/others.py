import os
from datetime import datetime


def get_next_batch_code(results_dir) -> str:
    folder_names = [
        folder for folder in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, folder))
    ]
    if folder_names:
        folder_names.sort()
        latest_folder_name = folder_names[-1]
        latest_batch_number = int(latest_folder_name.split("_")[0])
        next_batch_number = latest_batch_number + 1
    else:
        next_batch_number  = 1

    return f' {(next_batch_number):03d}'

def create_result_output_folder(results_dir="results") -> str:
    batch_code = get_next_batch_code(results_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    result_folder = f"{results_dir}/{batch_code}_{timestamp}/"
    result_folder_to_save_models = f"{results_dir}/{batch_code}_{timestamp}/models/"
    os.makedirs(result_folder_to_save_models, exist_ok=True)
    return result_folder