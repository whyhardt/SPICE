# Go over each study,
# then aggregate (mean loss) the results from each instance per condition (20 instances per condition)
# then compare these mean of each condition to each other, and output the best condition performance
# potentially also output the best instance per condition

import os
import json
import csv

cwd = os.getcwd() # this should be: /home/staff/i/isdelpozo/SPICE
STUDY = "weber2024" 
# weber2024, gru_weber2024
# kolff2025, gru_kolff2025


STUDY_PATH = os.path.join(cwd, f"weinhardt2026/slurm_training/runs/{STUDY}") # this should be: /home/staff/i/isdelpozo/SPICE/weinhardt2026/slurm_training/runs

aggregated_results = {}
iter=0

aggreg_type = "mean" # mean or median
if aggreg_type == "mean":
    print(f"Aggregating results for study: {STUDY} using mean")
elif aggreg_type == "median":
    print(f"Aggregating results for study: {STUDY} using median")

# iterate through each folder in runs/study
for instance in os.listdir(STUDY_PATH):
    iter+=1
    instance_keys = instance.split("_")
    # example: 000000_braun2018__benchmark__bench_braun2018_230647efb1__rep=0
    # ['000000', 'braun2018', '', 'benchmark', '', 'bench', 'braun2018', '230647efb1', '', 'rep=0']
    KEY = instance_keys[7] # this should be the unique identifier for the condition
    print(f"--- {iter}:Processing instance: {instance}, KEY: {KEY}")

    if KEY not in aggregated_results:
        print(f"--- {iter}: creating new dictionary for KEY: {KEY}")
        aggregated_results[KEY] = {
            "instances": [],
            "test_losses": [],
            "train_losses": [],
            "conditions": {},
            "best_instance": None,
            "median_instance": None,
            "worst_instance": None
        }

    # go into result.json
    with open(os.path.join(cwd, STUDY_PATH, instance, "result.json"), "r") as f:
        result_data = json.load(f)
        test_loss = result_data.get("test_loss", None)
        train_loss = result_data.get("train_loss", None)

        # save test_loss in dictionary under the corresponding KEY  
        aggregated_results[KEY]["instances"].append(instance)
        aggregated_results[KEY]["test_losses"].append(test_loss)
        aggregated_results[KEY]["train_losses"].append(train_loss)

    # go into run.json
    with open(os.path.join(cwd, STUDY_PATH, instance, "run.json"), "r") as f:       
        run_data = json.load(f)

        # save "training" dictionary under corresponding KEY
        aggregated_results[KEY]["conditions"] = run_data.get("training", {})
        aggregated_results[KEY]["conditions"].update(run_data.get("model", {}).get("init_kwargs", {})) 


# finalize each condition from the full set of collected instances
for key, value in aggregated_results.items():
    if not value["test_losses"]:
        continue

    sorted_pairs = sorted(zip(value["test_losses"], value["instances"], value["train_losses"]))
    best_pair = sorted_pairs[0]
    median_pair = sorted_pairs[len(sorted_pairs) // 2]
    worst_pair = sorted_pairs[-1]

    if aggreg_type == "mean":
        value["mean_test_loss"] = sum(value["test_losses"]) / len(value["test_losses"])
        value["mean_train_loss"] = sum(value["train_losses"]) / len(value["train_losses"])
    elif aggreg_type == "median":
        value["median_test_loss"] = median_pair[0]
        value["median_train_loss"] = median_pair[2]

    value["best_instance"] = best_pair[1]
    value["median_instance"] = median_pair[1]
    value["worst_instance"] = worst_pair[1]

    if aggreg_type == "mean":
        print(f"Condition: {key},\n --- Best test_loss: {best_pair[0]},\n ---  Worst test_loss: {worst_pair[0]},\n ---  Mean test_loss: {value['mean_test_loss']},\n ---  Mean train_loss: {value['mean_train_loss']}")
    elif aggreg_type == "median":
        print(f"Condition: {key},\n --- Best test_loss: {best_pair[0]},\n ---  Worst test_loss: {worst_pair[0]},\n ---  Median test_loss: {value['median_test_loss']},\n ---  Median train_loss: {value['median_train_loss']}")

# in the end, I should have a dictionary with all conditions, and the mean test_loss from each 20instances
# Now I would need a mechanism to return the best condition
overall_best_mean_loss = 1000
best_condition = None
best_instance = None
median_instance = None
worst_instance = None

if aggreg_type == "mean":
# Iterate through the dictionary
    for key, value in aggregated_results.items():
        mean_test_loss = value.get("mean_test_loss", None)
        if mean_test_loss is not None and mean_test_loss < overall_best_mean_loss:
            overall_best_mean_loss = mean_test_loss
            best_condition = key
            best_instance = aggregated_results[key].get("best_instance", None)
            median_instance = aggregated_results[key].get("median_instance", None)
            worst_instance = aggregated_results[key].get("worst_instance", None)
elif aggreg_type == "median":
    for key, value in aggregated_results.items():
        median_test_loss = value.get("median_test_loss", None)
        if median_test_loss is not None and median_test_loss < overall_best_mean_loss:
            overall_best_mean_loss = median_test_loss
            best_condition = key
            best_instance = aggregated_results[key].get("best_instance", None)
            median_instance = aggregated_results[key].get("median_instance", None)
            worst_instance = aggregated_results[key].get("worst_instance", None)

print(f"For study: {STUDY}")
print(f"Overall best mean test loss: {overall_best_mean_loss}")
print(f"Best condition: {best_condition}")
print(f"Best instance: {best_instance}")
print(f"Median instance: {median_instance}")
print(f"Worst instance: {worst_instance}")


# use csv file "best_conditions.csv"
# update csv file for the current study with the best condition and its mean test_loss (and potentially the best instance)

if aggreg_type == "mean":
    best_cond_file_path = os.path.join(cwd, "weinhardt2026/slurm_training/best_conditions_mean.csv")
elif aggreg_type == "median":
    best_cond_file_path = os.path.join(cwd, "weinhardt2026/slurm_training/best_conditions_median.csv")

fields = ["Study", "Best Condition", "Mean Test Loss", "Condition Details", "Best Instance", "Median Instance", "Worst Instance"]
row = {
    "Study": STUDY,
    "Best Condition": best_condition,
    "Mean Test Loss": overall_best_mean_loss,
    "Condition Details": aggregated_results[best_condition]["conditions"],
    "Best Instance": best_instance,
    "Median Instance": median_instance,
    "Worst Instance": worst_instance
}


# Open the file in write mode
with open(best_cond_file_path, mode='a', newline='') as file: #mode='w' for complete override of file, 'a' for appending
    # Create a csv.writer object
    writer = csv.DictWriter(file, fieldnames=fields)

    # Write data to the CSV file
    #writer.writeheader()
    writer.writerow(row)

# Print a confirmation message
print(f"CSV file '{best_cond_file_path}' written successfully.")