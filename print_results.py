import pandas as pd

results_dict = {
    "model": [],
    "logloss": [],
    "auc": []
}

if __name__ == "__main__":
    # get string of results
    with open("results.txt", "r") as file:
        for line in file.readlines():
            
            # update list of models
            if "results:" in line:
                results_dict["model"].append(line.split()[0])
            
            # update AUC
            elif "average_auc:" in line:
                results_dict["auc"].append(float(line.split()[1]))

            # update AUC
            elif "average_logloss:" in line:
                results_dict["logloss"].append(float(line.split()[1]))

            #print(line)
    #print(results_dict)
    print(pd.DataFrame(results_dict).set_index("model").to_csv())