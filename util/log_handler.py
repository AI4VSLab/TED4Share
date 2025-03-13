# --------------------------------------------------------
# log_hander to provide functions for logging the results
# Written by Sina Gholami from https://github.com/sinagh72/TED
# --------------------------------------------------------
import os


def log_results(log_root_dir, classes, results, dataset_name, comment):
    result = {}
    metrics = ["accuracy", "precision", "auc", "f1"]
    result["auc"] = results[0]["test_auc"]
    result["pr"] = results[0]["test_pr"]
    result["f1"] = results[0]["test_f1"]
    result["accuracy"] = results[0]["test_accuracy"]
    result["precision"] = results[0]["test_precision"]
    result["loss"] = results[0]["test_loss"]
    '''
    for c in classes.keys():
        for m in metrics:
            result[f"test_{m}_" + c] = results[0][f"test_{m}_" + c]
    '''
    result = {key: round(value, 4) for key, value in result.items()}
    dir_name = f"{log_root_dir}/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    log_name = f'{dir_name}/{dataset_name}_{comment}.txt'
    if os.path.exists(log_name):
        file_mode = "a+"
    else:
        file_mode = "w"
    separator_line = f"====================================\n"
    with open(log_name, file_mode) as f:
        if file_mode == "w":
            f.write(separator_line)  # Write the line if not found
        elif file_mode != "w":
            f.seek(0)  # Go to the beginning of the file to read
            content = f.read()  # Read the entire content of the file
            # Check if the specific line is already in the file
            if separator_line not in content:
                f.write(separator_line)  # Write the line if not found
        f.write(str(result))
        f.write('\n')
