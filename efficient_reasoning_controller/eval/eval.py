import os
import copy
import json
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing

from data_loader import ModelandTask
from method import *


# =========================================
# Configuration Area
# =========================================
MODELS = ["Qwen3-4B", "Qwen3-1.7B", "Qwen3-8B", "Qwen3-0.6B"] # "Qwen3-8B", "Qwen3-4B", 
DATASETS = ["hmmt25","aime25", "aime24"]
BOOTSTRAP = 1  # Set 1 to disable bootstrapping


# =========================================
# Method Configurations (new API)
# =========================================
method_configs = []

branch_groups = [2, 4, 8, 16, 32, 64]


thersholds = [0.95]
for branch in branch_groups:
    for threshold in thersholds:
        method_configs.append(ASCMethod(max_samples= branch, threshold=threshold))

window_sizes = [8]
for branch in branch_groups:
    for window in window_sizes:
        if window < branch:
            method_configs.append(ESCMethod(max_samples=branch, window_size=window))


#---- OptimalController methods ----
betas = [0, 0.2, 0.4, 0.5, 0.6,  0.8, 1.0]
for beta in betas:
    method_configs.append(OptimalController(config={"beta": beta}))

T_groups = [40]
# patiences = [4]
patiences = [7]
warm_ups = [15]
burst_freqs = [1]

# for branch in branch_groups:
#     method_configs.append(MajorityVoteMethod(max_samples=branch))

# # Parallel EST + only active branch v2 Strategies
for T in T_groups:
    for m in branch_groups:
        for prune_patience in patiences:
            for warm_up in warm_ups:
                method_configs.append(Parallel_Probe(num_chains=m, T=T, prune_patience=prune_patience, warm_up=warm_up))


# =========================================
# Global worker state
# =========================================
worker_task_instance = None


def init_worker(model_name, dataset_name):
    global worker_task_instance
    print(f"Worker process {os.getpid()} initializing task: {model_name} / {dataset_name}...")
    try:
        worker_task_instance = ModelandTask(model_name, dataset_name)
    except Exception as e:
        print(f"Error initializing worker {os.getpid()}: {e}")


def execute_eval_process(method_obj):
    """
    Evaluate one complete method in worker process.
    """
    global worker_task_instance
    if worker_task_instance is None:
        return None

    try:
        # defensively copy to avoid accidental shared-state mutation
        method_obj = copy.deepcopy(method_obj)

        result = worker_task_instance.evaluate(method_obj, bootstrap_iter=BOOTSTRAP)

        entry = {
            "Method": type(method_obj).__name__,
            "Acc": result["accuracy"],
            "Cost": int(result["avg_cost"]),
            "Std_Acc": result.get("std_accuracy", 0),
            "Description": method_obj.description(),
        }

        for k, v in method_obj.__dict__.items():
            if k == "trace":
                continue
            entry[k] = v

        return {
            "entry": entry,
            "traces": result.get("traces", []),
        }

    except Exception as e:
        print(f"Error evaluating {type(method_obj).__name__}: {e}")
        return None


def run_matrix_evaluation_multiprocess(model_name, dataset_name):
    jobs = [copy.deepcopy(m) for m in method_configs]

    print(f"Starting Matrix Eval ({len(jobs)} methods) with Multiprocessing...")

    DEBUG_MODE = False

    if DEBUG_MODE:
        print("!!! DEBUG MODE ACTIVE: Running sequentially in main process !!!")
        init_worker(model_name, dataset_name)
        results = []
        all_traces = []
        for method_obj in tqdm(jobs):
            res = execute_eval_process(method_obj)
            if res:
                results.append(res["entry"])
                traces = res.get("traces", [])
                if traces:
                    all_traces.extend({
                        "model": model_name,
                        "dataset": dataset_name,
                        "method": res["entry"].get("Method"),
                        "description": res["entry"].get("Description"),
                        **t,
                    } for t in traces)
        return pd.DataFrame(results), all_traces

    # 根据 CPU 数量和方法个数自动选择进程数，避免被 1 写死
    num_processes = min(multiprocessing.cpu_count(), len(jobs))
    results = []
    all_traces = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_processes,
        initializer=init_worker,
        initargs=(model_name, dataset_name),
    ) as executor:
        futures = {executor.submit(execute_eval_process, job): job for job in jobs}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(jobs)):
            try:
                res = future.result()
                if res:
                    results.append(res["entry"])
                    traces = res.get("traces", [])
                    if traces:
                        all_traces.extend({
                            "model": model_name,
                            "dataset": dataset_name,
                            "method": res["entry"].get("Method"),
                            "description": res["entry"].get("Description"),
                            **t,
                        } for t in traces)
            except Exception as exc:
                print(f"Job generated an exception: {exc}")

    return pd.DataFrame(results), all_traces


if __name__ == "__main__":
    for model in MODELS:
        for dataset in DATASETS:
            print(f"\nProcessing {model} on {dataset}...")
            try:
                data, traces = run_matrix_evaluation_multiprocess(model, dataset)

                output_dir = f"eval/test_results/matrix_results_{model}"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                data.to_csv(f"{output_dir}/{dataset}_raw_new_api.csv", index=False)
                print(f"Saved to {output_dir}/{dataset}_raw_new_api.csv")

                trace_path = f"{output_dir}/{dataset}_trace_new_api.jsonl"
                with open(trace_path, "w", encoding="utf-8") as f:
                    for row in traces:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"Saved traces to {trace_path}")

            except Exception as e:
                print(f"Error processing {model} / {dataset}: {e}")
