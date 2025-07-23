import os
import numpy as np
import pandas as pd

def method_command_config(method):
    if method == "CGA-dHash":
        return "--method CGA-dHash --cga_n_gen 100"
    elif method == "CGA-DFT":
        return "--method CGA-DFT --cga_n_gen 100"
    elif method == "aHash":
        return "--method aHash"
    elif method == "dHash":
        return "--method dHash"
    elif method == "MinMax":
        return "--method MinMax"
    elif method == "DFT":
        return "--method DFT"
    elif method == "ITQ":
        return "--method ITQ"
    elif method == "LBP":
        return "--method LBP"
    elif method == "LSH":
        return "--method LSH"
    elif method == "DHN":
        return "--method DHN"
    elif method == "DSH":
        return "--method DSH"
    elif method == "DTSH":
        return "--method DTSH"
    elif method == "CSQ":
        return "--method CSQ"
    elif method == "DPSH":
        return "--method DPSH"
    elif method == "Quantization":
        return "--method Quantization"
    else:
        raise ValueError(f"Method {method} not found")


if __name__ == "__main__":

    dataset = "tcga_pulmonary_densenet121"
    feature_selection = ""
    num_runs = 5

    methods = ["aHash", "dHash", "MinMax", "DFT", "ITQ", "LBP", "LSH", "DHN", "DSH", "DTSH", "CSQ", "DPSH", "Quantization", "CGA-dHash", "CGA-DFT"]
    
    for run in range(num_runs):
        random_seed = np.random.randint(0, 1000000)

        for method in methods:
            if os.path.exists(os.path.join("results", f"{dataset}_{feature_selection}_{method}.csv")):
                df = pd.read_csv(os.path.join("results", f"{dataset}_{feature_selection}_{method}.csv"))
                if df.shape[0] == num_runs:
                    print(f"Skipping {method} because all runs are completed")
                    continue

            command = f"python main.py"
            command += f" --seed {random_seed} "
            command += f" --dataset {dataset} "
            command += f" --download "
            command += f" --k 10 "
            command += method_command_config(method)
            os.system(command)

    comparison_df = pd.DataFrame()
    for method in methods:
        df = pd.read_csv(os.path.join("results", f"{dataset}_{feature_selection}_{method}.csv"))
        comparison_df[method] = df.mean(axis=0)
    comparison_df = comparison_df.transpose()
    comparison_df.columns = ["F1", "Precision@10", "mAP"]
    comparison_df["Method"] = methods
    comparison_df = comparison_df[["Method", "F1", "Precision@10", "mAP"]]
    # comparison_df = comparison_df.sort_values(by="F1", ascending=False)
    comparison_df.to_csv(os.path.join("results",f"{dataset}_{feature_selection}_summary.csv"), index=False)