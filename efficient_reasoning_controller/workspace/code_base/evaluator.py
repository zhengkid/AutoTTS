from data_loader import ModelandTask
from method import GreedySolver
from controller_api import MethodStrategyAdapter


def evaluate_strategy_instance(
    strategy,
    model_name: str,
    dataset_name: str,
    solver=None,
    bootstrap_iter: int = 1,
):
    task = ModelandTask(model_name, dataset_name)

    if solver is None:
        solver = GreedySolver()

    solver.branch_strategy = strategy
    result = task.evaluate(solver, bootstrap_iter=bootstrap_iter)

    return {
        "accuracy": result["accuracy"],
        "avg_total_tokens": result["avg_cost"],
        "avg_sequential_tokens": result.get("avg_sequential_cost", 0),
        "raw_result": result,
    }


def evaluate_method_instance(
    method,
    model_name: str,
    dataset_name: str,
    solver=None,
    bootstrap_iter: int = 1,
):
    strategy = MethodStrategyAdapter(method)
    return evaluate_strategy_instance(
        strategy=strategy,
        model_name=model_name,
        dataset_name=dataset_name,
        solver=solver,
        bootstrap_iter=bootstrap_iter,
    )