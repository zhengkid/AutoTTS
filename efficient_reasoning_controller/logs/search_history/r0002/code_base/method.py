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
    Speculative Confidence Rollout (SCR) Controller.

    Core idea — asymmetric depth allocation with plateau-triggered widening:

    Unlike all three seeds and Round-1 IBC, SCR allocates probe depth
    NON-UNIFORMLY across branches.  After warm-up it classifies each active
    (unfinished, unabandoned) branch as either "aligned" (its current answer
    matches the completed-pool winner) or "deviant" (it does not).

    Per round:
      - Aligned branches receive `burst_aligned` probe steps each  (> 1).
      - Deviant branches receive exactly 1 probe step.
      - This deepens agreeing branches faster, accelerating their completion
        and driving up the completed-pool confidence, while deviant branches
        still receive minimal probing (cheap, informative enough to decide
        whether to abandon).

    Abandonment: deviant branches whose `pool_disagree` counter reaches
    `abandon_patience` are dropped (same signal as IBC, anchored to the
    reliable completed pool, not noisy intermediate plurality).

    Plateau-triggered widening: the controller tracks the last
    `stall_patience` consecutive rounds in which the completed-pool
    confidence did NOT increase (a "confidence plateau").  When a plateau is
    detected AND budget remains, new branches are spawned aggressively
    (`burst_new` at once) rather than one at a time.  This switches from
    depth-first to breadth-first exactly when depth is not helping — a
    decision that IBC cannot make because it widens uniformly whenever
    confidence < threshold.

    Termination: same Beta-majority confidence gate over the completed-pool,
    triggered only after warm_up rounds AND min_complete finished branches.

    beta schedule (single knob in [0, 1]):
      beta = 0  -> small budget, tight confidence, low patience
      beta = 1  -> near full-budget, high confidence required, generous patience

    Novel mechanisms vs seeds and IBC:
      - ASC / ESC: full reads only; no mid-branch probe at all.
      - Parallel_Probe: uniform 1-step probing of all active chains together;
        no asymmetric burst; abandonment anchored to noisy intermediate
        plurality.
      - IBC: uniform 1-step probing per round; widen 1 branch/round when
        conf < thresh; no plateau detection; depth allocation is fixed.
      - SCR: non-uniform depth (aligned gets burst_aligned > 1, deviant gets
        1); plateau-triggered bulk widening (`burst_new` branches at once);
        these two mechanisms are genuinely absent from all prior proposals.
    """

    NAME = "optimal_controller"

    # Fixed structural constants
    _MAX_BRANCH   = 64
    _MAX_OUTER    = 600   # hard safety cap on outer iterations

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def _schedule(self, beta: float) -> dict:
        """
        Map scalar beta in [0, 1] to every internal hyperparameter.

        All values are non-decreasing in beta (more budget / patience as
        beta grows), using simple analytic (linear / clipped-linear) forms.

        warm_up          = round(2 + 8*b)          [2, 10]
          Rounds before confidence gate, abandonment, and burst fire.

        min_complete     = max(2, round(2 + 4*b))  [2, 6]
          Min finished branches before stop is allowed.

        conf_thresh      = 0.85 + 0.12*b           [0.85, 0.97]
          Beta-majority confidence required to stop.

        n_init           = max(2, round(2 + 6*b))  [2, 8]
          Initial simultaneous branches.

        max_branch_use   = round(4 + 60*b)         [4, 64]
          Total branch ceiling.

        abandon_patience = max(4, round(4 + 8*b))  [4, 12]
          Consecutive rounds of disagreement before abandonment.

        burst_aligned    = max(1, round(1 + 3*b))  [1, 4]
          Extra probe_more steps given to aligned branches per round.
          At beta=0 this is 1 (same as uniform); grows with beta.

        stall_patience   = max(2, round(2 + 6*b))  [2, 8]
          Consecutive rounds of flat confidence before plateau widening.

        burst_new        = max(1, round(1 + 3*b))  [1, 4]
          Number of new branches spawned in one plateau-widen event.
        """
        b = max(0.0, min(1.0, float(beta)))

        warm_up          = round(2 + 8 * b)
        min_complete     = max(2, round(2 + 4 * b))
        conf_thresh      = 0.85 + 0.12 * b
        n_init           = max(2, round(2 + 6 * b))
        max_branch_use   = min(self._MAX_BRANCH, round(4 + 60 * b))
        abandon_patience = max(4, round(4 + 8 * b))
        burst_aligned    = max(1, round(1 + 3 * b))
        stall_patience   = max(2, round(2 + 6 * b))
        burst_new        = max(1, round(1 + 3 * b))

        return {
            "warm_up":          warm_up,
            "min_complete":     min_complete,
            "conf_thresh":      conf_thresh,
            "n_init":           n_init,
            "max_branch_use":   max_branch_use,
            "abandon_patience": abandon_patience,
            "burst_aligned":    burst_aligned,
            "stall_patience":   stall_patience,
            "burst_new":        burst_new,
        }

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._beta        = float((config or {}).get("beta", 0.5))
        sched             = self._schedule(self._beta)
        self.warm_up          = sched["warm_up"]
        self.min_complete     = sched["min_complete"]
        self.conf_thresh      = sched["conf_thresh"]
        self.n_init           = sched["n_init"]
        self.max_branch_use   = sched["max_branch_use"]
        self.abandon_patience = sched["abandon_patience"]
        self.burst_aligned    = sched["burst_aligned"]
        self.stall_patience   = sched["stall_patience"]
        self.burst_new        = sched["burst_new"]
        self.trace_recorder   = MethodTraceRecorder()

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
        """Return (winner, confidence) over the completed-answer pool."""
        if not completed:
            return None, 0.0
        winner, top1, top2, _ = _vote_stats(completed)
        conf = _beta_majority_confidence(top1, top2)
        return winner, conf

    def _probe_branch(self, question, br: Dict[str, Any], steps: int,
                      completed_answers: List[str]) -> None:
        """
        Probe branch `br` for `steps` steps, updating state in-place.
        Appends to completed_answers if the branch finishes.
        """
        for _ in range(steps):
            if br["finished"] or br["abandoned"]:
                break
            out = _safe_probe_more(question, br["index"])
            if out is None:
                br["finished"] = True
                if br["latest_ans"] is not None:
                    completed_answers.append(br["latest_ans"])
                break
            new_ans, is_finish = out
            br["latest_ans"] = new_ans
            br["finished"] = is_finish
            if is_finish:
                completed_answers.append(new_ans)
                break

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, question) -> Optional[str]:
        self._reset_trace()
        self._trace_step(
            event="start",
            goal="initialize SCR run",
            step_input={"beta": self._beta},
            step_output="initialized",
            state={
                "warm_up":          self.warm_up,
                "min_complete":     self.min_complete,
                "conf_thresh":      round(self.conf_thresh, 4),
                "n_init":           self.n_init,
                "max_branch_use":   self.max_branch_use,
                "abandon_patience": self.abandon_patience,
                "burst_aligned":    self.burst_aligned,
                "stall_patience":   self.stall_patience,
                "burst_new":        self.burst_new,
            },
            decision="start speculative confidence rollout control",
        )

        # Branch records
        branches: List[Dict[str, Any]] = []
        completed_answers: List[str] = []
        total_spawned = 0

        # Plateau tracker
        prev_conf       = 0.0
        flat_rounds     = 0   # consecutive rounds conf did not improve

        # ---- Phase 0: open initial branches ----
        for _ in range(self.n_init):
            out = _safe_probe_new(question)
            if out is None:
                break
            ans, idx, is_finish = out
            total_spawned += 1
            br: Dict[str, Any] = {
                "index":         idx,
                "latest_ans":    ans,
                "finished":      is_finish,
                "abandoned":     False,
                "pool_disagree": 0,
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

            # Compute pool stats at the START of this round
            pool_winner, pool_conf = self._pool_stats(completed_answers)
            n_complete = len(completed_answers)

            # ---------- Asymmetric probing ----------
            # Classify active branches: aligned vs deviant
            aligned_brs = []
            deviant_brs = []
            for br in branches:
                if br["abandoned"] or br["finished"]:
                    continue
                if pool_winner is None or outer_step < self.warm_up:
                    # Before warm-up or no winner yet: treat all as aligned
                    aligned_brs.append(br)
                elif br["latest_ans"] == pool_winner:
                    aligned_brs.append(br)
                else:
                    deviant_brs.append(br)

            # Probe aligned branches with burst_aligned steps
            probe_steps_aligned = (
                self.burst_aligned if outer_step >= self.warm_up else 1
            )
            for br in aligned_brs:
                self._probe_branch(question, br, probe_steps_aligned,
                                   completed_answers)

            # Probe deviant branches with exactly 1 step
            for br in deviant_brs:
                self._probe_branch(question, br, 1, completed_answers)

            # Update pool stats after probing
            pool_winner, pool_conf = self._pool_stats(completed_answers)
            n_complete = len(completed_answers)

            # ---------- Abandon persistently-deviant branches ----------
            abandoned_this: List[int] = []
            if outer_step >= self.warm_up and pool_winner is not None:
                # Update pool_disagree counters
                for br in branches:
                    if br["abandoned"] or br["finished"]:
                        continue
                    if br["latest_ans"] != pool_winner:
                        br["pool_disagree"] += 1
                    else:
                        br["pool_disagree"] = 0

                # Abandon candidates; keep >= 2 unfinished
                n_unfinished = sum(
                    1 for br in branches
                    if not br["abandoned"] and not br["finished"]
                )
                cands = [
                    br for br in branches
                    if not br["abandoned"] and not br["finished"]
                    and br["pool_disagree"] >= self.abandon_patience
                ]
                max_abandon = max(0, n_unfinished - 2)
                cands_sorted = sorted(cands, key=lambda b: -b["pool_disagree"])
                for br in cands_sorted[:max_abandon]:
                    br["abandoned"] = True
                    abandoned_this.append(br["index"])

            n_active = sum(
                1 for br in branches
                if not br["abandoned"] and not br["finished"]
            )

            self._trace_step(
                event="forward",
                goal="asymmetric probe + abandonment",
                step_input={
                    "outer_step":           outer_step,
                    "pool_winner_before":   pool_winner,
                    "pool_conf_before":     round(prev_conf, 4),
                },
                step_output={
                    "n_aligned_probed":   len(aligned_brs),
                    "n_deviant_probed":   len(deviant_brs),
                    "total_completed":    n_complete,
                    "abandoned_this":     abandoned_this,
                    "pool_conf_after":    round(pool_conf, 4),
                },
                state={
                    "n_active":      n_active,
                    "total_spawned": total_spawned,
                },
                decision="check confidence gate",
            )

            # ---------- Confidence gate ----------
            self._trace_step(
                event="terminate_check",
                goal="check completed-pool confidence",
                step_input={
                    "outer_step":   outer_step,
                    "warm_up":      self.warm_up,
                    "conf_thresh":  round(self.conf_thresh, 4),
                    "min_complete": self.min_complete,
                },
                step_output={
                    "winner":     pool_winner,
                    "conf":       round(pool_conf, 4),
                    "n_complete": n_complete,
                },
                state={"total_spawned": total_spawned},
                decision="early stop if warm_up passed, min_complete met, conf >= thresh",
            )

            if (
                outer_step >= self.warm_up
                and n_complete >= self.min_complete
                and pool_conf >= self.conf_thresh
            ):
                self._trace_step(
                    event="finish",
                    goal="return final answer",
                    step_input={"outer_step": outer_step},
                    step_output={
                        "answer":      pool_winner,
                        "stop_reason": "confidence_threshold",
                        "conf":        round(pool_conf, 4),
                        "n_complete":  n_complete,
                    },
                    state={"total_spawned": total_spawned},
                    decision=f"early stop: conf >= thresh",
                )
                return pool_winner

            # ---------- Check all resolved ----------
            all_resolved = all(br["finished"] or br["abandoned"]
                               for br in branches)

            # ---------- Plateau detection + widening ----------
            conf_improved = pool_conf > prev_conf + 1e-6
            if conf_improved:
                flat_rounds = 0
            else:
                flat_rounds += 1

            plateau_hit = (
                flat_rounds >= self.stall_patience
                and outer_step >= self.warm_up
            )
            can_widen = total_spawned < self.max_branch_use

            # Determine how many new branches to open
            if can_widen and not all_resolved:
                if plateau_hit:
                    # Bulk widen: spawn burst_new branches
                    n_to_spawn = min(
                        self.burst_new,
                        self.max_branch_use - total_spawned,
                    )
                    flat_rounds = 0   # reset plateau counter after bulk widen
                elif pool_conf < self.conf_thresh and outer_step >= max(1, self.warm_up // 2):
                    # Incremental widen: 1 branch
                    n_to_spawn = 1
                else:
                    n_to_spawn = 0
            else:
                n_to_spawn = 0

            spawned_now = 0
            for _ in range(n_to_spawn):
                out = _safe_probe_new(question)
                if out is None:
                    break
                ans, idx, is_finish = out
                total_spawned += 1
                spawned_now += 1
                br_new: Dict[str, Any] = {
                    "index":         idx,
                    "latest_ans":    ans,
                    "finished":      is_finish,
                    "abandoned":     False,
                    "pool_disagree": 0,
                }
                branches.append(br_new)
                if is_finish:
                    completed_answers.append(ans)
                all_resolved = False   # new branch active

            self._trace_step(
                event="update_states",
                goal="plateau detection and widening",
                step_input={
                    "outer_step":   outer_step,
                    "conf_improved": conf_improved,
                    "flat_rounds":  flat_rounds,
                    "plateau_hit":  plateau_hit,
                },
                step_output={
                    "spawned_now":   spawned_now,
                    "total_spawned": total_spawned,
                    "all_resolved":  all_resolved,
                    "pool_conf":     round(pool_conf, 4),
                },
                state={"n_active": n_active},
                decision="continue or break",
            )

            prev_conf = pool_conf

            if all_resolved and spawned_now == 0:
                break

            outer_step += 1

        # ---- Final answer ----
        final_winner, final_conf = self._pool_stats(completed_answers)
        if final_winner is None:
            all_latest = [
                br["latest_ans"]
                for br in branches
                if not br["abandoned"] and br["latest_ans"] is not None
            ]
            final_winner = _majority_answer(all_latest)
            final_conf   = 0.0

        self._trace_step(
            event="finish",
            goal="return final answer",
            step_input={"outer_step": outer_step},
            step_output={
                "answer":        final_winner,
                "stop_reason":   "loop_end",
                "conf":          round(final_conf, 4),
                "n_complete":    len(completed_answers),
                "total_spawned": total_spawned,
            },
            state={"total_spawned": total_spawned},
            decision="majority of completed answers",
        )
        return final_winner