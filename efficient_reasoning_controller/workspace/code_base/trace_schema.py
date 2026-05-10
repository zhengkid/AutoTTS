def make_step_trace(
    step,
    majority_answer=None,
    vote_ratio=None,
    num_active=None,
    num_pruned=None,
    stable_cnt=None,
    stop_candidate=None,
    extra=None,
):
    return {
        "step": step,
        "majority_answer": majority_answer,
        "vote_ratio": vote_ratio,
        "num_active": num_active,
        "num_pruned": num_pruned,
        "stable_cnt": stable_cnt,
        "stop_candidate": stop_candidate,
        "extra": extra or {},
    }


def make_problem_trace(
    question_id,
    final_answer,
    gold_answer,
    correct,
    total_tokens,
    sequential_tokens,
    stop_reason,
    trace,
):
    return {
        "question_id": question_id,
        "final_answer": final_answer,
        "gold_answer": gold_answer,
        "correct": correct,
        "total_tokens": total_tokens,
        "sequential_tokens": sequential_tokens,
        "stop_reason": stop_reason,
        "trace": trace,
    }