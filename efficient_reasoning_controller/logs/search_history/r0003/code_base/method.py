import math
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
    Dual-Gate Confidence Control (DGCC).

    Core idea: two complementary stopping gates + lock-streak lazy probing.

    Gate 1 — Primary (strict): pool_conf >= conf_thresh
      Beta-majority confidence over the completed-answer pool (naturally-
      finished branches only).  Identical to IBC's primary signal.

    Gate 2 — Soft corroboration: pool_conf >= conf_thresh_soft AND
      active_align_rate >= act_thresh
      When the completed pool is moderately confident AND the overwhelming
      majority of ACTIVE (still-probing) branches already show the same
      answer as the pool winner, stop early.  The active branches act as
      low-cost corroborating witnesses that cross-validate the pool signal
      without needing them to complete.  This is absent from all seeds
      (which never look at partial answers for stopping) and from IBC/SCR
      (which only use completed branches in the gate).

    Lock-streak lazy probing (novel depth-allocation mechanism):
      Each active unfinished branch tracks `lock_streak` = consecutive
      probe rounds in which its answer did NOT change.  A branch with a
      high lock_streak is "sleeping" — it has converged to an intermediate
      answer and marginal probing is unlikely to flip it cheaply.  Such
      branches are probed only once every `sleep_period` rounds rather than
      every round.  This saves per-round token cost without discarding the
      branch or betting on it being wrong.  It is fundamentally different
      from SCR's asymmetric burst (which probes aligned branches MORE) and
      from IBC/parallel-probe (which probe all active branches every round).

    Widening — vote-gap proportional:
      Spawn ceil(gap_factor * (1 - vote_margin)) extra branches per round
      when confidence is below threshold, where vote_margin is the fraction
      of completed answers backing the leader.  A close race (margin near 0)
      triggers more widening than a near-threshold leader (margin near 1).
      This adapts width to uncertainty more tightly than a fixed 1-or-burst
      schedule.

    Abandonment — same completed-pool signal as IBC/SCR:
      Branches whose latest answer has disagreed with the pool winner for
      `abandon_patience` consecutive rounds are dropped (keeping >=2
      unfinished unless unavoidable).

    beta schedule (single knob in [0, 1]):
      All hyperparameters are smooth analytic functions of beta.
      beta=0 -> conservative, small budget, easy to stop;
      beta=1 -> near full-budget, hard to stop without strong evidence.

    Novel vs seeds:
      ASC/ESC: full reads only; no mid-branch probing at all.
      Parallel_Probe: no completed-pool; abandonment anchored to noisy
        intermediate plurality; no corroboration gate.
      IBC (r0001): single confidence gate; uniform 1-step probing per round;
        1-branch widening; no corroboration gate; no lazy probing.
      SCR (r0002): single confidence gate; burst probing for aligned branches
        (MORE steps, not LESS for locked ones); plateau-triggered bulk
        widening; no corroboration gate; no lock-streak tracking.
      DGCC: dual stopping gate (corroboration) + lock-streak lazy probing
        + vote-gap widening — three mechanisms absent from all prior work.
    """

    NAME = "optimal_controller"

    # Fixed structural constants (never tuned between runs)
    _MAX_BRANCH = 64
    _MAX_OUTER  = 600   # hard iteration safety cap

    # ------------------------------------------------------------------
    # Schedule — single beta knob maps to all hyperparameters
    # ------------------------------------------------------------------

    def _schedule(self, beta: float) -> dict:
        """
        Map scalar beta in [0, 1] to all internal hyperparameters.
        All schedules are non-decreasing in beta (more budget as beta grows).

        warm_up          = round(3 + 7*b)          [3, 10]
          Rounds before any stopping gate or abandonment fires.

        min_complete     = max(2, round(2 + 4*b))  [2, 6]
          Minimum completed branches before primary gate may fire.

        conf_thresh      = 0.85 + 0.10*b           [0.85, 0.95]
          Primary gate: Beta-majority confidence required to stop.
          Note: non-decreasing in beta means harder to stop at high beta.

        conf_thresh_soft = 0.60 + 0.15*b           [0.60, 0.75]
          Soft corroboration gate lower threshold.
          Also non-decreasing (harder soft gate at higher beta).

        act_thresh       = 0.70 + 0.10*b           [0.70, 0.80]
          Soft gate: minimum fraction of active branches showing pool winner.
          Non-decreasing (more active agreement required at higher beta).

        n_init           = max(2, round(2 + 6*b))  [2, 8]
          Branches opened simultaneously at startup.

        max_branch_use   = round(4 + 60*b)         [4, 64]
          Total branch budget ceiling.

        abandon_patience = max(4, round(4 + 8*b))  [4, 12]
          Consecutive rounds of disagreement with pool winner before abandon.

        sleep_period     = max(2, round(3 + 3*b))  [3, 6]
          Rounds between probes of a sleeping (locked) branch.
          Non-decreasing: at high beta probes are more reluctant to skip.

        lock_thresh      = max(3, round(4 + 2*b))  [4, 6]
          Lock-streak length required for a branch to enter sleep mode.
          Non-decreasing: higher beta = harder to enter sleep.

        gap_factor       = round(1 + 2*b)           [1, 3]
          Widening aggressiveness proportional to vote uncertainty.
        """
        b = max(0.0, min(1.0, float(beta)))

        warm_up          = round(3 + 7 * b)
        min_complete     = max(2, round(2 + 4 * b))
        conf_thresh      = 0.85 + 0.10 * b
        conf_thresh_soft = 0.60 + 0.15 * b
        act_thresh       = 0.70 + 0.10 * b
        n_init           = max(2, round(2 + 6 * b))
        max_branch_use   = min(self._MAX_BRANCH, round(4 + 60 * b))
        abandon_patience = max(4, round(4 + 8 * b))
        sleep_period     = max(2, round(3 + 3 * b))
        lock_thresh      = max(3, round(4 + 2 * b))
        gap_factor       = round(1 + 2 * b)

        return {
            "warm_up":          warm_up,
            "min_complete":     min_complete,
            "conf_thresh":      conf_thresh,
            "conf_thresh_soft": conf_thresh_soft,
            "act_thresh":       act_thresh,
            "n_init":           n_init,
            "max_branch_use":   max_branch_use,
            "abandon_patience": abandon_patience,
            "sleep_period":     sleep_period,
            "lock_thresh":      lock_thresh,
            "gap_factor":       gap_factor,
        }

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._beta           = float((config or {}).get("beta", 0.5))
        sched                = self._schedule(self._beta)
        self.warm_up         = sched["warm_up"]
        self.min_complete    = sched["min_complete"]
        self.conf_thresh     = sched["conf_thresh"]
        self.conf_thresh_soft = sched["conf_thresh_soft"]
        self.act_thresh      = sched["act_thresh"]
        self.n_init          = sched["n_init"]
        self.max_branch_use  = sched["max_branch_use"]
        self.abandon_patience = sched["abandon_patience"]
        self.sleep_period    = sched["sleep_period"]
        self.lock_thresh     = sched["lock_thresh"]
        self.gap_factor      = sched["gap_factor"]
        self.trace_recorder  = MethodTraceRecorder()

    # ------------------------------------------------------------------
    # Trace surface (matches seed style)
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

    def _pool_stats(self, completed: List[str]):
        """Return (winner, top1, top2, conf) over completed-answer pool."""
        if not completed:
            return None, 0, 0, 0.0
        winner, top1, top2, _ = _vote_stats(completed)
        conf = _beta_majority_confidence(top1, top2)
        return winner, top1, top2, conf

    def _active_align_rate(
        self,
        branches: List[Dict],
        pool_winner,
    ) -> float:
        """
        Fraction of active (unfinished, non-abandoned) branches whose
        latest answer matches pool_winner.  Returns 0.0 if no active branches
        or pool_winner is None.
        """
        if pool_winner is None:
            return 0.0
        active = [
            br for br in branches
            if not br["abandoned"] and not br["finished"]
        ]
        if not active:
            return 0.0
        matching = sum(1 for br in active if br["latest_ans"] == pool_winner)
        return matching / len(active)

    def _probe_one(
        self,
        question,
        br: Dict,
        completed_answers: List[str],
    ) -> None:
        """Probe branch br for one step; append to completed_answers if done."""
        out = _safe_probe_more(question, br["index"])
        if out is None:
            br["finished"] = True
            if br["latest_ans"] is not None:
                completed_answers.append(br["latest_ans"])
            return
        new_ans, is_finish = out
        if new_ans == br["latest_ans"]:
            br["lock_streak"] += 1
        else:
            br["lock_streak"] = 0
        br["latest_ans"] = new_ans
        br["finished"] = is_finish
        if is_finish:
            completed_answers.append(new_ans)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, question) -> Optional[str]:
        self._reset_trace()
        self._trace_step(
            event="start",
            goal="initialize DGCC run",
            step_input={"beta": self._beta},
            step_output="initialized",
            state={
                "warm_up":          self.warm_up,
                "min_complete":     self.min_complete,
                "conf_thresh":      round(self.conf_thresh, 4),
                "conf_thresh_soft": round(self.conf_thresh_soft, 4),
                "act_thresh":       round(self.act_thresh, 4),
                "n_init":           self.n_init,
                "max_branch_use":   self.max_branch_use,
                "abandon_patience": self.abandon_patience,
                "sleep_period":     self.sleep_period,
                "lock_thresh":      self.lock_thresh,
                "gap_factor":       self.gap_factor,
            },
            decision="start dual-gate confidence control",
        )

        # Branch state:
        #   index         : stable branch_index from probe_new
        #   latest_ans    : current intermediate (or final) answer
        #   finished      : bool — completed its full reasoning chain
        #   abandoned     : bool — dropped due to persistent disagreement
        #   lock_streak   : consecutive rounds with unchanged answer
        #   pool_disagree : consecutive rounds disagreeing with pool winner
        #   probe_age     : rounds since last probed (for sleep scheduling)
        branches: List[Dict[str, Any]] = []
        completed_answers: List[str] = []   # only from finished branches
        total_spawned = 0

        # ---- Phase 0: open n_init branches simultaneously ----
        for _ in range(self.n_init):
            out = _safe_probe_new(question)
            if out is None:
                break
            ans, idx, is_finish = out
            total_spawned += 1
            br: Dict[str, Any] = {
                "index":        idx,
                "latest_ans":   ans,
                "finished":     is_finish,
                "abandoned":    False,
                "lock_streak":  0,
                "pool_disagree": 0,
                "probe_age":    0,   # rounds since last probe
            }
            branches.append(br)
            if is_finish:
                completed_answers.append(ans)

        self._trace_step(
            event="init_branches",
            goal="open initial branch batch",
            step_input={"n_init": self.n_init},
            step_output={
                "n_spawned":   total_spawned,
                "n_completed": len(completed_answers),
            },
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

        # ---- Main loop ----
        outer_step = 0
        while outer_step < self._MAX_OUTER:

            # ---- Compute pool statistics ----
            pool_winner, top1, top2, pool_conf = self._pool_stats(completed_answers)
            n_complete = len(completed_answers)

            # ---- Probe active branches with lock-streak lazy scheduling ----
            # A sleeping branch (lock_streak >= lock_thresh) is only probed
            # once every sleep_period rounds.  All others are probed every round.
            for br in branches:
                if br["abandoned"] or br["finished"]:
                    continue
                sleeping = (
                    br["lock_streak"] >= self.lock_thresh
                    and outer_step >= self.warm_up
                )
                if sleeping:
                    # Probe only on multiples of sleep_period relative to when
                    # the branch first entered sleep (approximate with probe_age).
                    br["probe_age"] += 1
                    if br["probe_age"] % self.sleep_period != 0:
                        continue   # skip this round
                self._probe_one(question, br, completed_answers)

            # Update pool stats after this round of probing
            pool_winner, top1, top2, pool_conf = self._pool_stats(completed_answers)
            n_complete = len(completed_answers)

            # ---- Update lock_streak and pool_disagree counters ----
            if outer_step >= self.warm_up and pool_winner is not None:
                for br in branches:
                    if br["abandoned"] or br["finished"]:
                        continue
                    if br["latest_ans"] != pool_winner:
                        br["pool_disagree"] += 1
                    else:
                        br["pool_disagree"] = 0

            # ---- Abandon persistently-deviant branches ----
            abandoned_this: List[int] = []
            if outer_step >= self.warm_up and pool_winner is not None:
                n_unfinished = sum(
                    1 for br in branches
                    if not br["abandoned"] and not br["finished"]
                )
                cands = [
                    br for br in branches
                    if not br["abandoned"] and not br["finished"]
                    and br["pool_disagree"] >= self.abandon_patience
                ]
                # Keep at least 2 unfinished after abandonment
                max_abandon = max(0, n_unfinished - 2)
                cands_sorted = sorted(cands, key=lambda b: -b["pool_disagree"])
                for br in cands_sorted[:max_abandon]:
                    br["abandoned"] = True
                    abandoned_this.append(br["index"])

            n_active = sum(
                1 for br in branches
                if not br["abandoned"] and not br["finished"]
            )
            n_sleeping = sum(
                1 for br in branches
                if not br["abandoned"] and not br["finished"]
                and br["lock_streak"] >= self.lock_thresh
                and outer_step >= self.warm_up
            )

            self._trace_step(
                event="forward",
                goal="probe (with lazy sleep) + update counters + abandon",
                step_input={
                    "outer_step":  outer_step,
                    "pool_winner": pool_winner,
                    "pool_conf":   round(pool_conf, 4),
                },
                step_output={
                    "n_complete":    n_complete,
                    "n_active":      n_active,
                    "n_sleeping":    n_sleeping,
                    "abandoned_now": abandoned_this,
                },
                state={"total_spawned": total_spawned},
                decision="check dual-gate stop and widening",
            )

            # ---- Dual stopping gates (only after warm_up + min_complete) ----
            gates_eligible = (
                outer_step >= self.warm_up
                and n_complete >= self.min_complete
            )

            # Gate 1: primary — completed-pool confidence
            gate1 = gates_eligible and pool_conf >= self.conf_thresh

            # Gate 2: soft corroboration — moderate pool conf + high active align
            align_rate = self._active_align_rate(branches, pool_winner)
            gate2 = (
                gates_eligible
                and pool_conf >= self.conf_thresh_soft
                and align_rate >= self.act_thresh
                and pool_winner is not None
            )

            self._trace_step(
                event="terminate_check",
                goal="dual-gate confidence evaluation",
                step_input={
                    "outer_step":      outer_step,
                    "warm_up":         self.warm_up,
                    "conf_thresh":     round(self.conf_thresh, 4),
                    "conf_thresh_soft": round(self.conf_thresh_soft, 4),
                    "act_thresh":      round(self.act_thresh, 4),
                    "min_complete":    self.min_complete,
                },
                step_output={
                    "winner":      pool_winner,
                    "pool_conf":   round(pool_conf, 4),
                    "align_rate":  round(align_rate, 4),
                    "n_complete":  n_complete,
                    "gate1":       gate1,
                    "gate2":       gate2,
                },
                state={"total_spawned": total_spawned},
                decision="early stop if gate1 or gate2 fires",
            )

            if gate1 or gate2:
                stop_reason = "primary_confidence" if gate1 else "soft_corroboration"
                self._trace_step(
                    event="finish",
                    goal="return final answer",
                    step_input={"outer_step": outer_step},
                    step_output={
                        "answer":      pool_winner,
                        "stop_reason": stop_reason,
                        "pool_conf":   round(pool_conf, 4),
                        "align_rate":  round(align_rate, 4),
                        "n_complete":  n_complete,
                    },
                    state={"total_spawned": total_spawned},
                    decision=f"early stop via {stop_reason}",
                )
                return pool_winner

            # ---- Check if all branches resolved ----
            all_resolved = all(br["finished"] or br["abandoned"] for br in branches)

            # ---- Vote-gap proportional widening ----
            # vote_margin = top1 / n_complete in [0, 1], 0 if none.
            # n_to_spawn = ceil(gap_factor * (1 - vote_margin)), capped by budget.
            can_widen = total_spawned < self.max_branch_use
            want_widen = (
                can_widen
                and not all_resolved
                and outer_step >= max(1, self.warm_up // 2)
                and pool_conf < self.conf_thresh
            )

            spawned_now = 0
            if want_widen:
                if n_complete > 0:
                    vote_margin = top1 / n_complete
                else:
                    vote_margin = 0.0
                raw_spawn = math.ceil(
                    self.gap_factor * max(0.0, 1.0 - vote_margin)
                )
                n_to_spawn = min(
                    max(1, raw_spawn),
                    self.max_branch_use - total_spawned,
                )
                for _ in range(n_to_spawn):
                    out = _safe_probe_new(question)
                    if out is None:
                        break
                    ans, idx, is_finish = out
                    total_spawned += 1
                    spawned_now += 1
                    br_new: Dict[str, Any] = {
                        "index":        idx,
                        "latest_ans":   ans,
                        "finished":     is_finish,
                        "abandoned":    False,
                        "lock_streak":  0,
                        "pool_disagree": 0,
                        "probe_age":    0,
                    }
                    branches.append(br_new)
                    if is_finish:
                        completed_answers.append(ans)
                    all_resolved = False

            self._trace_step(
                event="update_states",
                goal="vote-gap widening snapshot",
                step_input={
                    "outer_step":  outer_step,
                    "want_widen":  want_widen,
                    "pool_conf":   round(pool_conf, 4),
                },
                step_output={
                    "spawned_now":   spawned_now,
                    "total_spawned": total_spawned,
                    "all_resolved":  all_resolved,
                },
                state={"n_active": n_active},
                decision="continue or terminate loop",
            )

            if all_resolved and spawned_now == 0:
                break

            outer_step += 1

        # ---- Final answer ----
        final_winner, _, _, final_conf = self._pool_stats(completed_answers)
        if final_winner is None:
            # Fallback: majority of all non-abandoned latest answers
            all_latest = [
                br["latest_ans"]
                for br in branches
                if not br["abandoned"] and br["latest_ans"] is not None
            ]
            final_winner = _majority_answer(all_latest)
            final_conf = 0.0

        self._trace_step(
            event="finish",
            goal="return final answer",
            step_input={"outer_step": outer_step},
            step_output={
                "answer":        final_winner,
                "stop_reason":   "loop_end",
                "pool_conf":     round(final_conf, 4),
                "n_complete":    len(completed_answers),
                "total_spawned": total_spawned,
            },
            state={"total_spawned": total_spawned},
            decision="majority of completed answers at loop end",
        )
        return final_winner