from abc import ABC, abstractmethod
from collections import Counter, deque
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import beta


# =========================================================
# New API
# =========================================================

class LLMDesignedMethod(ABC):
    """
    A complete reasoning-time method for the 2D probing environment.

    Each method is fully responsible for:
      - probing
      - stopping
      - pruning
      - voting
      - final answer selection
    """

    NAME = "unnamed_method"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def solve(self, question) -> Optional[str]:
        pass

    def __call__(self, question) -> Optional[str]:
        return self.solve(question)

    def description(self) -> str:
        if not self.config:
            return self.NAME
        return f"{self.NAME} | config={self.config}"


class MethodTraceRecorder:
    """
    Lightweight step-level trace recorder.
    """

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, **kwargs):
        self.steps.append(dict(kwargs))

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.steps)


# =========================================================
# Shared helpers
# =========================================================

def _majority_answer(answers: List[str]) -> Optional[str]:
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def _safe_probe_new(question):
    try:
        return question.probe_new()
    except (IndexError, ValueError):
        return None


def _safe_probe_more(question, index):
    try:
        return question.probe_more(index)
    except (IndexError, ValueError):
        return None


def _safe_full_read(question):
    """
    Old FullReadStrategy-compatible helper:
    consume one whole branch and return its final answer.
    """
    try:
        return question.get_new_branch_final_answer()
    except (IndexError, ValueError):
        return None


def _vote_stats(answers: List[str]):
    """
    Return (winner, top1, top2, total)
    """
    if not answers:
        return None, 0, 0, 0
    counts = Counter(answers)
    common = counts.most_common(2)
    winner = common[0][0]
    top1 = common[0][1]
    top2 = common[1][1] if len(common) > 1 else 0
    total = sum(counts.values())
    return winner, top1, top2, total


def _beta_majority_confidence(top1: int, top2: int) -> float:
    if top1 <= 0:
        return 0.0
    return float(1 - beta.cdf(0.5, top1 + 1, top2 + 1))


def _weighted_vote(pairs: List[tuple]) -> Optional[str]:
    if not pairs:
        return None
    counts: Dict[str, float] = {}
    for answer, weight in pairs:
        counts[answer] = counts.get(answer, 0.0) + weight
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

class ASCMethod(LLMDesignedMethod):
    """
    ASC method
    """
    NAME = "asc"

    def __init__(
        self,
        max_samples: int = 40,
        threshold: float = 0.95,
        config: Optional[Dict[str, Any]] = None,
    ):
        merged = {
            "max_samples": max_samples,
            "threshold": threshold,
        }
        if config:
            merged.update(config)
        super().__init__(merged)

        self.max_samples = max_samples
        self.threshold = threshold
        self.trace_recorder = MethodTraceRecorder()

    def _reset_trace(self) -> None:
        self.trace_recorder = MethodTraceRecorder()

    def _trace_step(
        self,
        *,
        event: str,
        goal: str,
        step_input: Dict[str, Any],
        step_output: Any,
        state: Dict[str, Any],
        decision: str,
    ) -> None:
        self.trace_recorder.add_step(
            event=event,
            goal=goal,
            input=step_input,
            output=step_output,
            state=state,
            decision=decision,
        )

    def get_last_trace(self) -> List[Dict[str, Any]]:
        return self.trace_recorder.to_list()

    def solve_with_trace(self, question) -> Dict[str, Any]:
        answer = self.solve(question)
        return {
            "answer": answer,
            "trace": self.get_last_trace(),
        }

    def _read_next_candidate(self, question, sample_idx: int, all_candidates: List[str]) -> Dict[str, Any]:
        ans = _safe_full_read(question)
        if ans is None:
            self._trace_step(
                event="read_stop",
                goal="read next candidate answer",
                step_input={"sample_idx": sample_idx},
                step_output=None,
                state={"all_candidates_n": len(all_candidates)},
                decision="stop due to branch exhaustion or read error",
            )
            return {"has_answer": False, "answer": None}

        all_candidates.append(ans)
        self._trace_step(
            event="read",
            goal="read next candidate answer",
            step_input={"sample_idx": sample_idx},
            step_output={"answer": ans},
            state={"all_candidates_n": len(all_candidates)},
            decision="continue until confidence threshold is met",
        )
        return {"has_answer": True, "answer": ans}

    def _evaluate_confidence(self, sample_idx: int, all_candidates: List[str]) -> Dict[str, Any]:
        if len(all_candidates) < 2:
            self._trace_step(
                event="confidence_skip",
                goal="compute confidence for early stop",
                step_input={"sample_idx": sample_idx, "all_candidates_n": len(all_candidates)},
                step_output=None,
                state={"all_candidates_n": len(all_candidates)},
                decision="need at least 2 samples to compare top votes",
            )
            return {"can_stop": False, "winner": None}

        counts = Counter(all_candidates)
        sorted_counts = counts.most_common(2)
        v1 = sorted_counts[0][1]
        v2 = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
        confidence = 1 - beta.cdf(0.5, v1 + 1, v2 + 1)
        winner = sorted_counts[0][0]
        self._trace_step(
            event="confidence_check",
            goal="compute confidence for early stop",
            step_input={"all_candidates": list(all_candidates)},
            step_output={
                "winner": winner,
                "counts": dict(counts),
                "top1": v1,
                "top2": v2,
                "confidence": float(confidence),
                "threshold": self.threshold,
            },
            state={"all_candidates_n": len(all_candidates)},
            decision="early stop if confidence > threshold",
        )

        if confidence > self.threshold:
            self._trace_step(
                event="finish",
                goal="return final answer",
                step_input={"all_candidates_n": len(all_candidates)},
                step_output={"answer": winner, "stop_reason": "confidence_threshold"},
                state={"all_candidates_n": len(all_candidates)},
                decision="stop early by confidence gate",
            )
            return {"can_stop": True, "winner": winner}

        return {"can_stop": False, "winner": winner}

    def _finalize(self, all_candidates: List[str]) -> Optional[str]:
        final_answer = _majority_answer(all_candidates)
        self._trace_step(
            event="finish",
            goal="return final answer",
            step_input={"all_candidates": list(all_candidates)},
            step_output={"answer": final_answer, "stop_reason": "max_samples_or_exhausted"},
            state={"all_candidates_n": len(all_candidates)},
            decision="fallback to majority vote",
        )
        return final_answer

    def solve(self, question) -> Optional[str]:
        self._reset_trace()
        self._trace_step(
            event="start",
            goal="initialize asc run",
            step_input={"question": repr(question)},
            step_output="initialized",
            state={"max_samples": self.max_samples, "threshold": self.threshold, "all_candidates_n": 0},
            decision="start sequential sampling with confidence stop",
        )
        all_candidates: List[str] = []

        for sample_idx in range(self.max_samples):
            read_result = self._read_next_candidate(question, sample_idx, all_candidates)
            if not read_result["has_answer"]:
                break

            conf_result = self._evaluate_confidence(sample_idx, all_candidates)
            if conf_result["can_stop"]:
                return conf_result["winner"]

        return self._finalize(all_candidates)


class ESCMethod(LLMDesignedMethod):
    """
    ESC method
    """
    NAME = "esc"

    def __init__(
        self,
        max_samples: int = 32,
        window_size: int = 5,
        config: Optional[Dict[str, Any]] = None,
    ):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")

        merged = {
            "max_samples": max_samples,
            "window_size": window_size,
        }
        if config:
            merged.update(config)
        super().__init__(merged)

        self.max_samples = max_samples
        self.window_size = window_size
        self.trace_recorder = MethodTraceRecorder()

    def _reset_trace(self) -> None:
        self.trace_recorder = MethodTraceRecorder()

    def _trace_step(
        self,
        *,
        event: str,
        goal: str,
        step_input: Dict[str, Any],
        step_output: Any,
        state: Dict[str, Any],
        decision: str,
    ) -> None:
        self.trace_recorder.add_step(
            event=event,
            goal=goal,
            input=step_input,
            output=step_output,
            state=state,
            decision=decision,
        )

    def get_last_trace(self) -> List[Dict[str, Any]]:
        return self.trace_recorder.to_list()

    def solve_with_trace(self, question) -> Dict[str, Any]:
        answer = self.solve(question)
        return {
            "answer": answer,
            "trace": self.get_last_trace(),
        }

    def _plan_windows(self) -> int:
        num_windows = self.max_samples // self.window_size
        self._trace_step(
            event="plan_windows",
            goal="compute sampling windows",
            step_input={
                "max_samples": self.max_samples,
                "window_size": self.window_size,
            },
            step_output={"num_windows": num_windows},
            state={"all_candidates_n": 0},
            decision="iterate fixed number of windows",
        )
        return num_windows

    def _collect_window(self, question, window_idx: int, all_candidates_n: int) -> List[str]:
        current_window: List[str] = []
        self._trace_step(
            event="window_start",
            goal="collect one sampling window",
            step_input={"window_idx": window_idx},
            step_output=None,
            state={"window_n": 0, "all_candidates_n": all_candidates_n},
            decision="read up to window_size full answers",
        )

        for read_idx in range(self.window_size):
            ans = _safe_full_read(question)
            if ans is None:
                self._trace_step(
                    event="window_read_stop",
                    goal="read full answer",
                    step_input={"window_idx": window_idx, "read_idx": read_idx},
                    step_output=None,
                    state={
                        "window_n": len(current_window),
                        "all_candidates_n": all_candidates_n,
                    },
                    decision="stop window because source exhausted/error",
                )
                break
            current_window.append(ans)
            self._trace_step(
                event="window_read",
                goal="read full answer",
                step_input={"window_idx": window_idx, "read_idx": read_idx},
                step_output={"answer": ans},
                state={
                    "window_n": len(current_window),
                    "all_candidates_n": all_candidates_n + len(current_window),
                },
                decision="continue reading current window",
            )

        return current_window

    def _evaluate_window(
        self,
        window_idx: int,
        current_window: List[str],
        all_candidates: List[str],
    ) -> Dict[str, Any]:
        """
        与种子 ESC.solve 中「空窗 continue / extend / 全窗一致则 return」顺序一致。
        早停条件必须与种子相同：len(set(current_window)) == 1。
        """
        if not current_window:
            self._trace_step(
                event="window_empty",
                goal="evaluate window",
                step_input={"window_idx": window_idx},
                step_output=None,
                state={"window_n": 0, "all_candidates_n": len(all_candidates)},
                decision="skip empty window",
            )
            return {"continue": True, "early_answer": None}

        all_candidates.extend(current_window)
        unanimous = len(set(current_window)) == 1
        self._trace_step(
            event="window_evaluate",
            goal="check early stop by unanimity",
            step_input={"window_answers": list(current_window)},
            step_output={
                "window_candidates": list(current_window),
                "unanimous": unanimous,
            },
            state={
                "window_n": len(current_window),
                "all_candidates_n": len(all_candidates),
            },
            decision="if unanimous then return current_window[0] else continue",
        )

        if unanimous:
            final_answer = current_window[0]
            self._trace_step(
                event="finish",
                goal="return final answer",
                step_input={
                    "all_candidates_n": len(all_candidates),
                    "stop_window_idx": window_idx,
                },
                step_output={
                    "answer": final_answer,
                    "stop_reason": "window_unanimity",
                },
                state={
                    "all_candidates_n": len(all_candidates),
                    "windows_used": window_idx + 1,
                },
                decision="early stop: unanimous window",
            )
            return {"continue": False, "early_answer": final_answer}

        return {"continue": True, "early_answer": None}

    def _finalize(self, all_candidates: List[str], num_windows: int) -> Optional[str]:
        final_answer = _majority_answer(all_candidates)
        self._trace_step(
            event="finish",
            goal="return final answer",
            step_input={"all_candidates": list(all_candidates)},
            step_output={
                "answer": final_answer,
                "stop_reason": "max_windows_or_exhausted",
            },
            state={"all_candidates_n": len(all_candidates), "windows_used": num_windows},
            decision="fallback to majority vote",
        )
        return final_answer

    def solve(self, question) -> Optional[str]:
        self._reset_trace()
        self._trace_step(
            event="start",
            goal="initialize esc run",
            step_input={"question": repr(question)},
            step_output="initialized",
            state={
                "max_samples": self.max_samples,
                "window_size": self.window_size,
                "all_candidates_n": 0,
            },
            decision="start windowed sampling",
        )
        all_candidates: List[str] = []
        num_windows = self._plan_windows()

        for window_idx in range(num_windows):
            current_window = self._collect_window(
                question, window_idx, len(all_candidates)
            )
            eval_result = self._evaluate_window(
                window_idx, current_window, all_candidates
            )
            if not eval_result["continue"]:
                return eval_result["early_answer"]

        return self._finalize(all_candidates, num_windows)


class Parallel_Probe(LLMDesignedMethod):
    """
    Parallel-Probe method
    """
    NAME = "Parallel_Probe"

    def __init__(
        self,
        num_chains: int = 64,
        K: int = 5,
        T: int = 14,
        eps_inter: float = 5.0,
        eps_intra: float = 5.0,
        prune_patience: int = 8,
        warm_up: int = 10,
        probe_burst: int = 1,
        max_steps: int = 100,
        config: Optional[Dict[str, Any]] = None,
    ):
        merged = {
            "num_chains": num_chains,
            "K": K,
            "T": T,
            "eps_inter": eps_inter,
            "eps_intra": eps_intra,
            "prune_patience": prune_patience,
            "warm_up": warm_up,
            "probe_burst": probe_burst,
            "max_steps": max_steps,
        }
        if config:
            merged.update(config)
        super().__init__(merged)

        self.num_chains = num_chains
        self.K = K
        self.T = T
        self.eps_inter = eps_inter
        self.eps_intra = eps_intra
        self.prune_patience = prune_patience
        self.warm_up = warm_up
        self.probe_burst = probe_burst
        self.max_steps = max_steps
        self.trace_recorder = MethodTraceRecorder()

    def _reset_trace(self) -> None:
        self.trace_recorder = MethodTraceRecorder()

    def _trace_step(
        self,
        *,
        event: str,
        goal: str,
        step_input: Dict[str, Any],
        step_output: Any,
        state: Dict[str, Any],
        decision: str,
    ) -> None:
        self.trace_recorder.add_step(
            event=event,
            goal=goal,
            input=step_input,
            output=step_output,
            state=state,
            decision=decision,
        )

    def get_last_trace(self) -> List[Dict[str, Any]]:
        return self.trace_recorder.to_list()

    def solve_with_trace(self, question) -> Dict[str, Any]:
        answer = self.solve(question)
        return {
            "answer": answer,
            "trace": self.get_last_trace(),
        }

    def _init_branches(self, question) -> Dict[str, Any]:
        branches: List[Dict[str, Any]] = []
        histories: List[List[Any]] = []
        off_track_counts: List[int] = []

        for _ in range(self.num_chains):
            out = _safe_probe_new(question)
            if out is None:
                break
            ans, idx, is_finish = out
            branches.append({
                "index": idx,
                "finished": is_finish,
                "pruned": False,
            })
            histories.append([ans])
            off_track_counts.append(0)

        return {
            "branches": branches,
            "histories": histories,
            "off_track_counts": off_track_counts,
        }

    def _forward_branches(self, question, branches, histories) -> Dict[str, Any]:
        active_indices: List[int] = []
        active_answers: List[Any] = []

        for i, branch in enumerate(branches):
            if branch["pruned"]:
                continue

            if not branch["finished"]:
                last_ans = histories[i][-1]
                for _ in range(self.probe_burst):
                    if branch["finished"]:
                        break
                    nxt = _safe_probe_more(question, branch["index"])
                    if nxt is None:
                        branch["finished"] = True
                        break
                    ans, is_finish = nxt
                    last_ans = ans
                    branch["finished"] = is_finish
                histories[i].append(last_ans)

            active_indices.append(i)
            active_answers.append(histories[i][-1])

        return {
            "active_indices": active_indices,
            "active_answers": active_answers,
        }

    def _update_states(self, step, branches, histories, off_track_counts, active_indices, active_answers) -> Dict[str, Any]:
        if not active_indices:
            return {
                "has_active": False,
                "active_indices": active_indices,
                "active_answers": active_answers,
                "step": step,
                "off_track_counts": off_track_counts,
            }

        winner_ans = _majority_answer(active_answers)

        if step >= self.warm_up and len(active_indices) > 1:
            for i in active_indices:
                branch = branches[i]
                if branch["finished"]:
                    continue
                if histories[i][-1] != winner_ans:
                    off_track_counts[i] += 1
                else:
                    off_track_counts[i] = 0

        return {
            "has_active": True,
            "winner_ans": winner_ans,
            "active_indices": active_indices,
            "active_answers": active_answers,
            "step": step,
            "off_track_counts": off_track_counts,
        }

    def _prune_phase(self, branches, histories, updated_state, prev_winner, stable_cnt) -> Dict[str, Any]:
        if not updated_state["has_active"]:
            return {
                "break_loop": True,
                "has_active": False,
                "winner_ans": prev_winner,
                "active_indices": [],
                "active_answers": [],
                "stable_cnt": stable_cnt,
                "new_prev_winner": prev_winner,
                "early_return": False,
                "return_answer": None,
            }

        winner_ans = updated_state["winner_ans"]
        step = updated_state["step"]
        off_track_counts = updated_state["off_track_counts"]
        active_indices = list(updated_state["active_indices"])
        active_answers = list(updated_state["active_answers"])
        prune_candidates: List[int] = []

        if step >= self.warm_up and len(active_indices) > 1:
            for i in active_indices:
                branch = branches[i]
                if branch["finished"]:
                    continue
                if off_track_counts[i] >= self.prune_patience:
                    prune_candidates.append(i)

        for i in prune_candidates:
            branches[i]["pruned"] = True

        if prune_candidates:
            active_indices = [i for i in active_indices if not branches[i]["pruned"]]
            active_answers = [histories[i][-1] for i in active_indices]
            if not active_indices:
                return {
                    "break_loop": True,
                    "has_active": False,
                    "winner_ans": prev_winner,
                    "active_indices": [],
                    "active_answers": [],
                    "stable_cnt": stable_cnt,
                    "new_prev_winner": prev_winner,
                    "early_return": False,
                    "return_answer": None,
                }
            winner_ans = _majority_answer(active_answers)

        return {
            "break_loop": False,
            "has_active": True,
            "winner_ans": winner_ans,
            "active_indices": active_indices,
            "active_answers": active_answers,
            "stable_cnt": stable_cnt,
            "new_prev_winner": prev_winner,
            "early_return": False,
            "return_answer": None,
        }

    def _termination_phase(self, branches, phase_state, prev_winner, stable_cnt) -> Dict[str, Any]:
        if phase_state["break_loop"] and not phase_state["has_active"]:
            return phase_state

        winner_ans = phase_state["winner_ans"]

        if winner_ans == prev_winner:
            stable_cnt += 1
        else:
            stable_cnt = 0

        if stable_cnt >= self.T:
            return {
                "break_loop": False,
                "stable_cnt": stable_cnt,
                "new_prev_winner": winner_ans,
                "early_return": True,
                "return_answer": winner_ans,
            }

        if all(b["finished"] or b["pruned"] for b in branches):
            return {
                "break_loop": True,
                "stable_cnt": stable_cnt,
                "new_prev_winner": winner_ans,
                "early_return": False,
                "return_answer": None,
            }

        return {
            "break_loop": False,
            "stable_cnt": stable_cnt,
            "new_prev_winner": winner_ans,
            "early_return": False,
            "return_answer": None,
        }


    def solve(self, question) -> Optional[str]:
        self._reset_trace()
        self._trace_step(
            event="start",
            goal="initialize parallel probe run",
            step_input={"question": repr(question)},
            step_output="initialized",
            state={
                "num_chains": self.num_chains,
                "K": self.K,
                "T": self.T,
                "warm_up": self.warm_up,
                "prune_patience": self.prune_patience,
                "probe_burst": self.probe_burst,
                "max_steps": self.max_steps,
            },
            decision="spawn chains then loop forward / majority / prune / terminate",
        )

        init_state = self._init_branches(question)
        branches = init_state["branches"]
        histories = init_state["histories"]
        off_track_counts = init_state["off_track_counts"]

        self._trace_step(
            event="init_branches",
            goal="spawn initial probe branches",
            step_input={"num_chains_requested": self.num_chains},
            step_output={
                "n_branches": len(branches),
                "branches": [
                    {"index": b["index"], "finished": b["finished"], "pruned": b["pruned"]}
                    for b in branches
                ],
            },
            state={"n_branches": len(branches)},
            decision="continue main loop if any branch exists",
        )

        if not branches:
            self._trace_step(
                event="finish",
                goal="return final answer",
                step_input={},
                step_output={"answer": None, "stop_reason": "no_branches"},
                state={"step": 0},
                decision="probe_new returned no chains",
            )
            return None

        stable_cnt = 0
        prev_winner = None
        step = 0
        finished_by_break = False

        while step < self.max_steps:
            self._trace_step(
                event="iteration_start",
                goal="main loop tick",
                step_input={"step": step},
                step_output=None,
                state={
                    "stable_cnt": stable_cnt,
                    "prev_winner": prev_winner,
                    "n_branches": len(branches),
                },
                decision="forward all active branches",
            )

            forward_state = self._forward_branches(question, branches, histories)
            active_indices = forward_state["active_indices"]
            active_answers = forward_state["active_answers"]

            self._trace_step(
                event="forward",
                goal="probe_more burst on unfinished branches",
                step_input={"step": step},
                step_output={
                    "active_indices": list(active_indices),
                    "active_answers": list(active_answers),
                    "branch_snapshot": [
                        {
                            "i": i,
                            "index": branches[i]["index"],
                            "finished": branches[i]["finished"],
                            "pruned": branches[i]["pruned"],
                            "last_answer": histories[i][-1],
                        }
                        for i in range(len(branches))
                    ],
                },
                state={"n_active": len(active_indices)},
                decision="majority vote among active last answers",
            )

            updated_state = self._update_states(
                step=step,
                branches=branches,
                histories=histories,
                off_track_counts=off_track_counts,
                active_indices=active_indices,
                active_answers=active_answers,
            )
            self._trace_step(
                event="update_states",
                goal="majority winner and off-track counters",
                step_input={"step": step, "warm_up": self.warm_up},
                step_output={
                    "has_active": updated_state["has_active"],
                    "winner_ans": updated_state.get("winner_ans"),
                    "off_track_counts": list(updated_state["off_track_counts"]),
                },
                state={"n_active": len(active_indices)},
                decision="increment off_track when disagreeing with majority after warm_up",
            )

            pruned_before = [branches[i]["pruned"] for i in range(len(branches))]
            pruned_state = self._prune_phase(
                branches=branches,
                histories=histories,
                updated_state=updated_state,
                prev_winner=prev_winner,
                stable_cnt=stable_cnt,
            )
            newly_pruned = [
                i
                for i in range(len(branches))
                if branches[i]["pruned"] and not pruned_before[i]
            ]
            self._trace_step(
                event="prune",
                goal="prune branches past patience vs majority",
                step_input={"step": step, "prune_patience": self.prune_patience},
                step_output={
                    "break_loop": pruned_state["break_loop"],
                    "has_active": pruned_state["has_active"],
                    "winner_ans": pruned_state["winner_ans"],
                    "active_indices_after": list(pruned_state["active_indices"]),
                    "newly_pruned_indices": newly_pruned,
                },
                state={"n_active_after": len(pruned_state["active_indices"])},
                decision="drop chains with high off_track; may clear all active",
            )

            decision = self._termination_phase(
                branches=branches,
                phase_state=pruned_state,
                prev_winner=prev_winner,
                stable_cnt=stable_cnt,
            )
            self._trace_step(
                event="terminate_check",
                goal="stable winner or all finished",
                step_input={"T": self.T, "prev_winner": prev_winner},
                step_output={
                    "stable_cnt": decision["stable_cnt"],
                    "new_prev_winner": decision["new_prev_winner"],
                    "early_return": decision["early_return"],
                    "break_loop": decision["break_loop"],
                    "return_answer": decision.get("return_answer"),
                },
                state={"step": step},
                decision="early exit if stable_cnt>=T; break if all finished/pruned",
            )

            stable_cnt = decision["stable_cnt"]
            prev_winner = decision["new_prev_winner"]

            if decision["early_return"]:
                ans = decision["return_answer"]
                self._trace_step(
                    event="finish",
                    goal="return final answer",
                    step_input={"step": step},
                    step_output={"answer": ans, "stop_reason": "stable_majority_T"},
                    state={"stable_cnt": stable_cnt, "prev_winner": prev_winner},
                    decision="consensus stable for T steps",
                )
                return ans

            if decision["break_loop"]:
                self._trace_step(
                    event="finish",
                    goal="return final answer",
                    step_input={"step": step},
                    step_output={
                        "answer": prev_winner,
                        "stop_reason": "all_branches_done_or_no_active",
                    },
                    state={"stable_cnt": stable_cnt, "prev_winner": prev_winner},
                    decision="exit loop then return last majority winner",
                )
                finished_by_break = True
                break

            step += 1

        if not finished_by_break:
            self._trace_step(
                event="finish",
                goal="return final answer",
                step_input={"step": step, "max_steps": self.max_steps},
                step_output={"answer": prev_winner, "stop_reason": "max_steps"},
                state={"stable_cnt": stable_cnt, "prev_winner": prev_winner},
                decision="hit max_steps without early stable stop",
            )
        return prev_winner



# =========================================================
# Your proposed OptimalController method here
# =========================================================

class OptimalController(LLMDesignedMethod):
    """
    Speculative Depth-Gated Voting (SDGV).

    Key idea:
      Each branch earns a "maturity" score based on how many consecutive probe
      steps it has produced the same answer (its "lock streak"). Branches that
      quickly converge to a stable answer are treated as more trustworthy and
      receive higher weight in the final vote.

    Controller loop:
      1. Maintain a cohort of active branches; probe each until it "locks"
         (answer unchanged for `lock_streak` consecutive steps) or finishes.
      2. After each round of probing, compute a depth-weighted Beta-majority
         confidence over all non-pruned branches (locked branches weighted by
         their maturity, unlocked ones weighted at 1.0).
      3. If confidence exceeds `conf_thresh`, terminate early.
      4. If confidence is still low after `warm_up` rounds, widen by spawning
         new branches (up to `max_branch` total).
      5. Optionally prune branches whose answer persistently differs from the
         current weighted-majority winner for `prune_patience` rounds after
         they have locked (confident minority branches).
      6. At termination, return the depth-weighted majority answer.

    Beta schedule (single knob):
      All internal hyperparameters are deterministic functions of beta in [0,1].
      beta=0  -> very conservative (few branches, low weight boost, high thresh)
      beta=1  -> near-full budget (more branches, deeper exploration, lower thresh)

    Novelty vs seeds:
      - Unlike ASC/ESC: uses incremental probing (not full reads), so intermediate
        answer trajectories inform trust weighting.
      - Unlike Parallel_Probe: pruning is gated on individual branch *lock status*
        (convergence to stable answer) rather than on disagreement with a global
        majority from the start; depth allocation is non-uniform.
      - The depth-weighted vote is a new aggregation not present in any seed.
    """

    NAME = "optimal_controller"

    # Fixed structural constants (not tuned between runs)
    _MAX_BRANCH = 64
    _MAX_OUTER_STEPS = 200  # hard cap on outer loop iterations

    def _schedule(self, beta: float) -> dict:
        """
        Map a single scalar beta in [0,1] to all internal hyperparameters.
        All non-decreasing in beta (more budget / less aggressive stopping).

        Analytic forms:
          warm_up          = round(2 + 6 * beta)           -> [2, 8]
          lock_streak      = max(1, round(2 + 2 * beta))   -> [2, 4]
          prune_patience   = max(2, round(3 + 5 * beta))   -> [3, 8]
          conf_thresh      = 0.98 - 0.14 * beta            -> [0.84, 0.98]
              (higher beta -> lower threshold -> easier to stop)
              Wait - we want beta=1 more aggressive spending. Lower thresh means
              EASIER to stop, which is LESS budget. So flip:
              conf_thresh  = 0.98 - 0.14 * (1 - beta)     -> [0.84 at beta=0, 0.98 at beta=1]
              Actually: higher thresh = harder to stop early = more budget.
              So conf_thresh should be NON-DECREASING in beta to use more budget.
              conf_thresh  = 0.85 + 0.13 * beta            -> [0.85, 0.98]
          max_cohort       = round(4 + 28 * beta)          -> [4, 32]  (initial + expand budget)
          maturity_boost   = 1.0 + 3.0 * beta              -> [1.0, 4.0]
          min_locked_frac  = 0.6 - 0.2 * beta              -> [0.4, 0.6] (fraction of cohort
                             that must be locked before conf check is meaningful)
              Higher beta -> lower min_locked_frac -> can stop with fewer locks
              Wait, we want beta=1 to spend more. More locked_frac needed means
              harder to stop early, which uses more budget. So:
              min_locked_frac = 0.4 + 0.3 * beta           -> [0.4, 0.7]
        """
        b = float(beta)
        b = max(0.0, min(1.0, b))

        warm_up = round(2 + 6 * b)                          # [2, 8]
        lock_streak = max(1, round(2 + 2 * b))              # [2, 4]
        prune_patience = max(2, round(3 + 5 * b))           # [3, 8]
        conf_thresh = 0.85 + 0.13 * b                       # [0.85, 0.98]
        max_cohort = round(4 + 28 * b)                      # [4, 32]
        maturity_boost = 1.0 + 3.0 * b                     # [1.0, 4.0]
        min_locked_frac = 0.4 + 0.3 * b                    # [0.4, 0.70]

        return {
            "warm_up": warm_up,
            "lock_streak": lock_streak,
            "prune_patience": prune_patience,
            "conf_thresh": conf_thresh,
            "max_cohort": max_cohort,
            "maturity_boost": maturity_boost,
            "min_locked_frac": min_locked_frac,
        }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        beta_val = (config or {}).get("beta", 0.5)
        self._beta = float(beta_val)
        sched = self._schedule(self._beta)
        self.warm_up = sched["warm_up"]
        self.lock_streak = sched["lock_streak"]
        self.prune_patience = sched["prune_patience"]
        self.conf_thresh = sched["conf_thresh"]
        self.max_cohort = min(sched["max_cohort"], self._MAX_BRANCH)
        self.maturity_boost = sched["maturity_boost"]
        self.min_locked_frac = sched["min_locked_frac"]
        self.trace_recorder = MethodTraceRecorder()

    # ------------------------------------------------------------------
    # Trace helpers (matching seed surface)
    # ------------------------------------------------------------------

    def _reset_trace(self) -> None:
        self.trace_recorder = MethodTraceRecorder()

    def _trace_step(
        self,
        *,
        event: str,
        goal: str,
        step_input: Dict[str, Any],
        step_output: Any,
        state: Dict[str, Any],
        decision: str,
    ) -> None:
        self.trace_recorder.add_step(
            event=event,
            goal=goal,
            input=step_input,
            output=step_output,
            state=state,
            decision=decision,
        )

    def get_last_trace(self) -> List[Dict[str, Any]]:
        return self.trace_recorder.to_list()

    def solve_with_trace(self, question) -> Dict[str, Any]:
        answer = self.solve(question)
        return {"answer": answer, "trace": self.get_last_trace()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_weighted_confidence(
        self,
        branches: List[Dict[str, Any]],
    ) -> tuple:
        """
        Compute depth-weighted majority vote and Beta-confidence.

        Each non-pruned branch contributes a vote with weight:
          locked branch  -> maturity_boost  (answer converged)
          unlocked       -> 1.0
        Returns (winner, confidence, top1_weight, top2_weight).
        """
        weighted: Dict[str, float] = {}
        for br in branches:
            if br["pruned"]:
                continue
            ans = br["current_answer"]
            if ans is None:
                continue
            w = self.maturity_boost if br["locked"] else 1.0
            weighted[ans] = weighted.get(ans, 0.0) + w

        if not weighted:
            return None, 0.0, 0.0, 0.0

        sorted_items = sorted(weighted.items(), key=lambda x: -x[1])
        winner = sorted_items[0][0]
        top1 = sorted_items[0][1]
        top2 = sorted_items[1][1] if len(sorted_items) > 1 else 0.0

        # Approximate integer counts for Beta CDF
        total_weight = sum(weighted.values())
        # Normalize to integer-ish counts (scale to sum ~total_branches)
        n_active = sum(1 for br in branches if not br["pruned"])
        if total_weight > 0 and n_active > 0:
            scale = n_active / total_weight
        else:
            scale = 1.0
        top1_int = max(1, round(top1 * scale))
        top2_int = max(0, round(top2 * scale))

        conf = _beta_majority_confidence(top1_int, top2_int)
        return winner, conf, top1, top2

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, question) -> Optional[str]:
        self._reset_trace()
        self._trace_step(
            event="start",
            goal="initialize SDGV run",
            step_input={"beta": self._beta},
            step_output="initialized",
            state={
                "warm_up": self.warm_up,
                "lock_streak": self.lock_streak,
                "prune_patience": self.prune_patience,
                "conf_thresh": self.conf_thresh,
                "max_cohort": self.max_cohort,
                "maturity_boost": self.maturity_boost,
                "min_locked_frac": self.min_locked_frac,
            },
            decision="start speculative depth-gated voting",
        )

        # branches: list of dicts tracking per-branch state
        # Each branch dict:
        #   index           : branch_index from probe_new
        #   current_answer  : latest probe answer
        #   finished        : bool, branch exhausted
        #   pruned          : bool
        #   streak          : consecutive same-answer count
        #   locked          : bool (streak >= lock_streak)
        #   disagree_rounds : rounds disagreeing with weighted winner after lock
        branches: List[Dict[str, Any]] = []
        total_spawned = 0

        # ---- Phase 1: spawn initial cohort ----
        init_size = max(2, self.max_cohort // 4) if self.max_cohort >= 4 else 2
        for _ in range(init_size):
            if total_spawned >= self._MAX_BRANCH:
                break
            out = _safe_probe_new(question)
            if out is None:
                break
            ans, idx, is_finish = out
            total_spawned += 1
            branches.append({
                "index": idx,
                "current_answer": ans,
                "prev_answer": None,
                "finished": is_finish,
                "pruned": False,
                "streak": 1,
                "locked": is_finish or (1 >= self.lock_streak),
                "disagree_rounds": 0,
            })

        self._trace_step(
            event="init_branches",
            goal="spawn initial cohort",
            step_input={"init_size": init_size},
            step_output={"n_spawned": len(branches)},
            state={"total_spawned": total_spawned},
            decision="proceed to main loop",
        )

        if not branches:
            self._trace_step(
                event="finish",
                goal="return final answer",
                step_input={},
                step_output={"answer": None, "stop_reason": "no_branches"},
                state={"total_spawned": 0},
                decision="no branches available",
            )
            return None

        outer_step = 0

        while outer_step < self._MAX_OUTER_STEPS:
            # ---- Probe all non-pruned, non-finished, non-locked branches ----
            probed_this_round = 0
            for br in branches:
                if br["pruned"] or br["finished"] or br["locked"]:
                    continue
                out = _safe_probe_more(question, br["index"])
                if out is None:
                    br["finished"] = True
                    br["locked"] = True
                    continue
                new_ans, is_finish = out
                probed_this_round += 1
                prev = br["current_answer"]
                if new_ans == prev:
                    br["streak"] += 1
                else:
                    br["streak"] = 1
                br["prev_answer"] = prev
                br["current_answer"] = new_ans
                br["finished"] = is_finish
                if br["streak"] >= self.lock_streak or is_finish:
                    br["locked"] = True

            # ---- Compute weighted confidence ----
            winner, conf, top1_w, top2_w = self._compute_weighted_confidence(branches)

            n_active = sum(1 for br in branches if not br["pruned"])
            n_locked = sum(1 for br in branches if not br["pruned"] and br["locked"])
            locked_frac = n_locked / n_active if n_active > 0 else 0.0

            self._trace_step(
                event="update_states",
                goal="probe non-locked branches and compute confidence",
                step_input={"outer_step": outer_step},
                step_output={
                    "winner": winner,
                    "conf": round(conf, 4),
                    "n_active": n_active,
                    "n_locked": n_locked,
                    "locked_frac": round(locked_frac, 3),
                    "probed_this_round": probed_this_round,
                },
                state={"total_spawned": total_spawned},
                decision="check confidence threshold and pruning",
            )

            # ---- Prune locked branches persistently disagreeing with winner ----
            if outer_step >= self.warm_up and winner is not None and n_active > 1:
                for br in branches:
                    if br["pruned"] or not br["locked"]:
                        continue
                    if br["current_answer"] != winner:
                        br["disagree_rounds"] += 1
                        if br["disagree_rounds"] >= self.prune_patience:
                            br["pruned"] = True
                    else:
                        br["disagree_rounds"] = 0

                # Recompute after pruning
                winner, conf, top1_w, top2_w = self._compute_weighted_confidence(branches)
                n_active = sum(1 for br in branches if not br["pruned"])
                n_locked = sum(1 for br in branches if not br["pruned"] and br["locked"])
                locked_frac = n_locked / n_active if n_active > 0 else 0.0

            self._trace_step(
                event="prune",
                goal="prune persistently disagreeing locked branches",
                step_input={"outer_step": outer_step, "prune_patience": self.prune_patience},
                step_output={
                    "n_active_after": n_active,
                    "winner": winner,
                    "conf": round(conf, 4),
                },
                state={"total_spawned": total_spawned},
                decision="widen or terminate based on confidence",
            )

            # ---- Early termination check ----
            all_done = all(br["pruned"] or br["locked"] for br in branches)
            enough_locked = locked_frac >= self.min_locked_frac and n_locked >= 2

            if outer_step >= self.warm_up and enough_locked and conf >= self.conf_thresh:
                self._trace_step(
                    event="finish",
                    goal="return final answer",
                    step_input={"outer_step": outer_step},
                    step_output={
                        "answer": winner,
                        "stop_reason": "confidence_threshold",
                        "conf": round(conf, 4),
                        "n_locked": n_locked,
                    },
                    state={"total_spawned": total_spawned},
                    decision=f"early stop: conf={conf:.3f} >= {self.conf_thresh}",
                )
                return winner

            if all_done and n_active == 0:
                # All pruned — return prev winner
                self._trace_step(
                    event="finish",
                    goal="return final answer",
                    step_input={"outer_step": outer_step},
                    step_output={
                        "answer": winner,
                        "stop_reason": "all_pruned",
                    },
                    state={"total_spawned": total_spawned},
                    decision="all branches pruned",
                )
                return winner

            # ---- Widen: spawn more branches if confidence insufficient ----
            can_widen = total_spawned < self.max_cohort and total_spawned < self._MAX_BRANCH
            want_widen = (
                can_widen
                and outer_step >= max(1, self.warm_up // 2)
                and conf < self.conf_thresh
                and (not all_done)
            )

            if want_widen:
                # Spawn one new branch per widen event (incremental)
                out = _safe_probe_new(question)
                if out is not None:
                    ans, idx, is_finish = out
                    total_spawned += 1
                    branches.append({
                        "index": idx,
                        "current_answer": ans,
                        "prev_answer": None,
                        "finished": is_finish,
                        "pruned": False,
                        "streak": 1,
                        "locked": is_finish or (1 >= self.lock_streak),
                        "disagree_rounds": 0,
                    })

                self._trace_step(
                    event="forward",
                    goal="widen cohort",
                    step_input={"outer_step": outer_step, "want_widen": True},
                    step_output={
                        "total_spawned": total_spawned,
                        "max_cohort": self.max_cohort,
                    },
                    state={"conf": round(conf, 4)},
                    decision="add branch to improve coverage",
                )
            else:
                self._trace_step(
                    event="forward",
                    goal="widen cohort",
                    step_input={"outer_step": outer_step, "want_widen": False},
                    step_output={"total_spawned": total_spawned},
                    state={"conf": round(conf, 4)},
                    decision="no widening: budget/confidence condition not met",
                )

            # ---- Check if all branches are done (locked/finished/pruned) ----
            if all_done:
                # Nothing left to probe; break and return final vote
                self._trace_step(
                    event="terminate_check",
                    goal="all branches locked or pruned",
                    step_input={"outer_step": outer_step},
                    step_output={"all_done": True},
                    state={"total_spawned": total_spawned},
                    decision="exit loop",
                )
                break

            outer_step += 1

        # ---- Final vote ----
        winner, conf, _, _ = self._compute_weighted_confidence(branches)
        self._trace_step(
            event="finish",
            goal="return final answer",
            step_input={"outer_step": outer_step},
            step_output={
                "answer": winner,
                "stop_reason": "loop_end",
                "conf": round(conf, 4),
                "total_spawned": total_spawned,
            },
            state={"total_spawned": total_spawned},
            decision="return depth-weighted majority answer",
        )
        return winner