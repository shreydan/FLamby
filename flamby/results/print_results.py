import os
import numpy as np
import pandas as pd
import re

dir_path = './'
results = []
dataset_names = []
dirs_multiple_seeds = [
#     os.path.join(dir_path, "results_benchmark_fed_camelyon16"),
#     os.path.join(dir_path, "results_benchmark_fed_lidc_idri"),
    os.path.join(dir_path, "results_benchmark_fed_cifar10"),
#     os.path.join(dir_path, "results_benchmark_fed_tcga_brca"),
   os.path.join(dir_path, "results_benchmark_fed_kits19"),
#	os.path.join(dir_path, "results_benchmark_fed_ixi"),
#     os.path.join(dir_path, "results_benchmark_fed_isic2019"),
#     os.path.join(dir_path, "results_benchmark_fed_heart_disease"),
#     os.path.join(dir_path, "results_benchmark_fed_covid19"),
]
for dir in dirs_multiple_seeds:
    csv_files = [os.path.join(dir, f) for f in os.listdir(dir)]
    result_pds = [pd.read_csv(f) for f in csv_files if os.path.isfile(f)]
    df = pd.concat(result_pds, ignore_index=True)
    results.append(df)
    dataset_names.append("_".join(dir.split("/")[-1].split(".")[0].split("_")[2:]))


res = results[0]
res = res.loc[res["Test"] != "Pooled Test"]
strategies_names = [
    "FedAvg",
    "Scaffold",
    "FedProx",
    "Cyclic",
    "FedAdagrad",
    "FedYogi",
    "FedAdam",
#     "FedAvgFineTuning"
]
# Filtering only 100 updates strategies
strategies = [strat + str(100) for strat in strategies_names]
current_methods = (
        ["Pooled Training"]
        + ["Local " + str(i) for i in range(10)]
        + strategies
    )
res = res.loc[res["Method"].isin(current_methods)]
res = res.rename(columns={"Method": "Training Method"})
res.loc[res["Training Method"] == "Pooled Training", "Training Method"] = "Pooled"
    
res["Training Method"] = [
        re.sub("Local", "Client", m) for m in res["Training Method"].tolist()
    ]


print(res.groupby('Training Method')['Metric'].mean())
