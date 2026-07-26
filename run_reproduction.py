#!/usr/bin/env python3
"""Bounded LiveCodeBench reproduction of Test-Time Scaling via Error Localization.

The four torchrun ranks are independent seeds on four GPUs. Rank zero downloads
the public model and V6 data once, then gathers every seed's measurements and
prints a compact JSON evidence record for OpenResearch.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import pickle
import re
import resource
import subprocess
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".run-cache"
DATA_URL = (
    "https://huggingface.co/datasets/livecodebench/"
    "code_generation_lite/resolve/main/test6.jsonl"
)
SYSTEM = (
    "You are an expert competitive programmer. Solve the problem in Python 3. "
    "Reason carefully, then give one complete solution inside a ```python code fence."
)
GENERIC_FEEDBACK = "Your previous attempt was unsuccessful."
NULL_INSTRUCTION = (
    "Repeat the previous attempt word by word, but skip portions of redundant thinking."
)


def setup_distributed() -> tuple[int, int, torch.device]:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, torch.device(f"cuda:{local_rank}")


def download_assets(rank: int, model_name: str) -> tuple[Path, Path]:
    model_dir = CACHE / "model"
    data_path = CACHE / "test6.jsonl"
    if rank == 0:
        CACHE.mkdir(parents=True, exist_ok=True)
        if not data_path.exists():
            response = requests.get(DATA_URL, timeout=120)
            response.raise_for_status()
            data_path.write_bytes(response.content)
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            token=os.environ.get("HF_TOKEN") or None,
        )
    dist.barrier()
    return model_dir, data_path


def decode_tests(value: str) -> list[dict[str, Any]]:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        unpacked = zlib.decompress(base64.b64decode(value.encode("utf-8")))
        try:
            return json.loads(unpacked.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            restored = pickle.loads(unpacked)
            return json.loads(restored) if isinstance(restored, str) else restored


def load_tasks(path: Path, per_difficulty: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    eligible: list[dict[str, Any]] = []
    for row in rows:
        public = decode_tests(row["public_test_cases"])
        private = decode_tests(row["private_test_cases"])
        tests = public + private
        if not tests or any(t.get("testtype") != "stdin" for t in tests):
            continue
        row["_public"] = public
        row["_private"] = private
        eligible.append(row)

    selected: list[dict[str, Any]] = []
    for difficulty in ("easy", "medium", "hard"):
        group = [r for r in eligible if r["difficulty"].lower() == difficulty]
        group.sort(
            key=lambda r: hashlib.sha256(r["question_id"].encode()).hexdigest()
        )
        selected.extend(group[:per_difficulty])
    if len(selected) != 3 * per_difficulty:
        raise RuntimeError(
            f"Expected {3 * per_difficulty} stratified stdin tasks, got {len(selected)}"
        )
    return selected


def build_problem(task: dict[str, Any]) -> str:
    starter = task.get("starter_code", "")
    suffix = f"\n\nStarter code:\n```python\n{starter}\n```" if starter else ""
    return (
        f"{task['question_content']}{suffix}\n\n"
        "Return a complete Python 3 solution that reads standard input and writes "
        "standard output."
    )


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.S | re.I)
    if blocks:
        return blocks[-1].strip()
    marker = text.rfind("```")
    if marker >= 0:
        tail = text[marker + 3 :]
        return re.sub(r"^(?:python|py)\s*", "", tail, flags=re.I).strip()
    return text.strip()


def _limit_process() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))


def normalize_output(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def outputs_match(actual: str, expected: str) -> bool:
    actual_norm = normalize_output(actual)
    expected_norm = normalize_output(expected)
    if actual_norm == expected_norm:
        return True
    actual_tokens = actual_norm.split()
    expected_tokens = expected_norm.split()
    if len(actual_tokens) != len(expected_tokens):
        return False
    try:
        actual_numbers = np.asarray([float(token) for token in actual_tokens])
        expected_numbers = np.asarray([float(token) for token in expected_tokens])
    except ValueError:
        return False
    return bool(np.allclose(actual_numbers, expected_numbers, rtol=1e-5, atol=1e-6))


def execute_case(code: str, case: dict[str, Any]) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="ttel-exec-") as temp_dir:
        source = Path(temp_dir) / "solution.py"
        source.write_text(code)
        try:
            proc = subprocess.run(
                ["python", "-I", str(source)],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=7,
                cwd=temp_dir,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                preexec_fn=_limit_process,
            )
        except subprocess.TimeoutExpired:
            return False, "Timed out"
        except Exception as exc:
            return False, f"Execution error: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        message = proc.stderr.strip().splitlines()
        return False, "Runtime error: " + (message[-1] if message else str(proc.returncode))
    actual = normalize_output(proc.stdout)
    expected = normalize_output(case["output"])
    if not outputs_match(actual, expected):
        return False, f"Wrong answer. Expected {expected[:240]!r}; observed {actual[:240]!r}"
    return True, "Passed"


def evaluate_candidate(
    text: str, task: dict[str, Any], feedback_kind: str
) -> tuple[bool, bool, str]:
    code = extract_code(text)
    if not code:
        return False, False, "No Python solution was found."
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError as exc:
        return False, False, f"Syntax error: {exc.msg} at line {exc.lineno}"

    public_ok = True
    public_message = "Passed public tests but failed hidden tests."
    for case in task["_public"]:
        ok, message = execute_case(code, case)
        if not ok:
            public_ok = False
            public_message = message
            break
    if public_ok:
        for case in task["_private"]:
            ok, _ = execute_case(code, case)
            if not ok:
                return False, True, public_message
        return True, True, "All tests passed."
    if feedback_kind == "environment":
        return False, False, public_message
    return False, False, GENERIC_FEEDBACK


def prompt_ids(tokenizer: Any, problem: str, history: str | None = None) -> torch.Tensor:
    user = problem
    if history:
        user += "\n\n" + history
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
    )


def history_text(attempt: str, feedback: str, null: bool = False) -> str:
    instruction = (
        NULL_INSTRUCTION
        if null
        else "Solve the original question. Use the feedback to make changes."
    )
    return (
        "### Attempt:\nThe following is a previous attempt to solve the question:\n"
        f"{attempt}\n\n### Feedback:\nThe following is feedback from your "
        f"unsuccessful earlier attempt:\n{feedback}\n\n{instruction}"
    )


@torch.inference_mode()
def generate(
    model: Any,
    tokenizer: Any,
    base_ids: torch.Tensor,
    prefix: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    full_input = torch.cat([base_ids.to(device), prefix.to(device)], dim=1)
    output = model.generate(
        full_input,
        do_sample=True,
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        max_new_tokens=int(cfg["max_new_tokens"]),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    continuation = output[:, full_input.shape[1] :]
    trajectory = torch.cat([prefix.to(device), continuation], dim=1)
    return trajectory, int(continuation.shape[1])


@torch.inference_mode()
def token_probabilities(
    model: Any,
    context: torch.Tensor,
    trajectory: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    context = context.to(device)
    trajectory = trajectory.to(device)
    inputs = torch.cat([context, trajectory], dim=1)
    logits = model(inputs).logits
    start = context.shape[1] - 1
    positions = logits[:, start : start + trajectory.shape[1], :]
    chosen = trajectory.unsqueeze(-1)
    probs = torch.softmax(positions.float(), dim=-1).gather(-1, chosen).squeeze()
    result = probs.detach().cpu().numpy().astype(np.float32)
    del logits, positions, probs
    torch.cuda.empty_cache()
    return np.atleast_1d(result)


def prefix_syntax_valid(tokenizer: Any, prefix: torch.Tensor) -> bool:
    text = tokenizer.decode(prefix[0], skip_special_tokens=True)
    code = extract_code(text)
    if "```" not in text and not code.lstrip().startswith(
        ("import ", "from ", "def ", "class ", "#")
    ):
        return True
    lines = code.splitlines()
    for end in range(len(lines), max(-1, len(lines) - 4), -1):
        fragment = "\n".join(lines[:end]).strip()
        if not fragment:
            return True
        try:
            ast.parse(fragment)
            return True
        except SyntaxError:
            continue
    return False


def syntax_adjust(tokenizer: Any, trajectory: torch.Tensor, index: int) -> int:
    floor = max(1, index - 96)
    for candidate in range(index, floor - 1, -1):
        decoded = tokenizer.decode(
            trajectory[0, :candidate], skip_special_tokens=True
        )
        if decoded.endswith(("\n", "```", ". ", ": ")):
            return candidate
    return index


def localize(
    model: Any,
    tokenizer: Any,
    student_context: torch.Tensor,
    true_context: torch.Tensor,
    null_context: torch.Tensor,
    trajectory: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    student = token_probabilities(model, student_context, trajectory, device)
    teacher = token_probabilities(model, true_context, trajectory, device)
    baseline = token_probabilities(model, null_context, trajectory, device)
    delta_true = student - teacher
    delta_null = student - baseline
    no_null = cfg["method"] == "ttel_no_null"
    if no_null:
        candidates = np.flatnonzero(delta_true > float(cfg["tau"]))
        scores = delta_true
    else:
        candidates = np.flatnonzero(
            (delta_true > float(cfg["tau"]))
            & (delta_null <= float(cfg["tau_baseline"]))
        )
        scores = baseline - teacher
    raw_index = int(candidates[np.argmax(scores[candidates])]) if len(candidates) else -1
    prefix_len = raw_index
    if raw_index >= 0 and cfg.get("branch_mode") == "syntax":
        prefix_len = syntax_adjust(tokenizer, trajectory, raw_index)
    prefix = trajectory[:, : max(0, prefix_len)]
    return {
        "spike_count": int(len(candidates)),
        "raw_branch_index": raw_index,
        "branch_index": int(prefix_len),
        "branch_fraction": (
            float(prefix_len / max(1, trajectory.shape[1])) if raw_index >= 0 else None
        ),
        "prefix_valid": prefix_syntax_valid(tokenizer, prefix) if raw_index >= 0 else None,
        "max_filtered_score": (
            float(scores[candidates].max()) if len(candidates) else None
        ),
        "prefix": prefix.detach().cpu(),
    }


def run_task(
    task: dict[str, Any],
    model: Any,
    tokenizer: Any,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    problem = build_problem(task)
    method = cfg["method"]
    prefix = torch.empty((1, 0), dtype=torch.long)
    current_history: str | None = None
    attempts: list[dict[str, Any]] = []
    cumulative_tokens = 0
    solved = False

    for turn in range(int(cfg["attempts"])):
        if method == "independent":
            current_history = None
            prefix = torch.empty((1, 0), dtype=torch.long)
        student_context = prompt_ids(tokenizer, problem, current_history)
        trajectory, new_tokens = generate(
            model, tokenizer, student_context, prefix, cfg, device
        )
        cumulative_tokens += new_tokens
        text = tokenizer.decode(trajectory[0], skip_special_tokens=True)
        success, public_ok, feedback = evaluate_candidate(
            text, task, cfg["feedback"]
        )
        record: dict[str, Any] = {
            "turn": turn + 1,
            "success": bool(success),
            "public_ok": bool(public_ok),
            "new_tokens": new_tokens,
            "cumulative_tokens": cumulative_tokens,
            "trajectory_tokens": int(trajectory.shape[1]),
        }
        if success:
            solved = True
            attempts.append(record)
            break

        if method == "refinement":
            answer = extract_code(text)
            current_history = history_text(answer, GENERIC_FEEDBACK)
            prefix = torch.empty((1, 0), dtype=torch.long)
        elif method.startswith("ttel"):
            true_feedback = feedback if cfg["feedback"] == "environment" else GENERIC_FEEDBACK
            true_history = history_text(text, true_feedback)
            null_history = history_text(text, NULL_INSTRUCTION, null=True)
            true_context = prompt_ids(tokenizer, problem, true_history)
            null_context = prompt_ids(tokenizer, problem, null_history)
            diagnostic = localize(
                model,
                tokenizer,
                student_context,
                true_context,
                null_context,
                trajectory,
                cfg,
                device,
            )
            prefix = diagnostic.pop("prefix")
            current_history = true_history
            if diagnostic["raw_branch_index"] < 0:
                prefix = torch.empty((1, 0), dtype=torch.long)
                current_history = None
            record.update(diagnostic)
        attempts.append(record)

    return {
        "question_id": task["question_id"],
        "difficulty": task["difficulty"],
        "solved": solved,
        "attempts": attempts,
        "total_generated_tokens": cumulative_tokens,
    }


def summarize(seed_results: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    tasks = [task for seed in seed_results for task in seed["tasks"]]
    curve = []
    for budget in range(1, int(cfg["attempts"]) + 1):
        successes = []
        tokens = []
        for task in tasks:
            visible = task["attempts"][:budget]
            successes.append(any(a["success"] for a in visible))
            tokens.append(sum(a["new_tokens"] for a in visible))
        curve.append(
            {
                "attempt_budget": budget,
                "pass_rate": float(np.mean(successes)),
                "mean_generated_tokens": float(np.mean(tokens)),
            }
        )
    diagnostic_attempts = [
        attempt
        for task in tasks
        for attempt in task["attempts"]
        if "spike_count" in attempt
    ]
    branches = [
        attempt for attempt in diagnostic_attempts if attempt["raw_branch_index"] >= 0
    ]
    return {
        "method": cfg["method"],
        "feedback": cfg["feedback"],
        "branch_mode": cfg["branch_mode"],
        "model": cfg["model"],
        "n_seeds": len(seed_results),
        "n_tasks_per_seed": len(seed_results[0]["tasks"]),
        "curve": curve,
        "final_pass_rate": curve[-1]["pass_rate"],
        "mean_total_generated_tokens": curve[-1]["mean_generated_tokens"],
        "mean_spikes": (
            float(np.mean([a["spike_count"] for a in diagnostic_attempts]))
            if diagnostic_attempts
            else None
        ),
        "branch_rate": (
            float(len(branches) / len(diagnostic_attempts))
            if diagnostic_attempts
            else None
        ),
        "mean_branch_fraction": (
            float(np.mean([a["branch_fraction"] for a in branches]))
            if branches
            else None
        ),
        "prefix_valid_rate": (
            float(np.mean([a["prefix_valid"] for a in branches])) if branches else None
        ),
        "task_ids": [task["question_id"] for task in seed_results[0]["tasks"]],
    }


def main() -> None:
    started = time.time()
    cfg = json.loads((ROOT / "config.json").read_text())
    rank, world, device = setup_distributed()
    if world != 4:
        raise RuntimeError(f"Protocol requires exactly four ranks/GPUs, got {world}")
    if rank == 0:
        print("CONFIG_JSON=" + json.dumps(cfg, sort_keys=True), flush=True)
        print(
            "COMPUTE_JSON="
            + json.dumps(
                {
                    "backend": "kubernetes",
                    "gpu_model": torch.cuda.get_device_name(0),
                    "gpu_count": world,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    model_dir, data_path = download_assets(rank, cfg["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    tasks = load_tasks(data_path, int(cfg["tasks_per_difficulty"]))
    seed = int(cfg["seeds"][rank])
    set_seed(seed)
    task_results = []
    for index, task in enumerate(tasks):
        result = run_task(task, model, tokenizer, cfg, device)
        task_results.append(result)
        print(
            "PROGRESS_JSON="
            + json.dumps(
                {
                    "rank": rank,
                    "seed": seed,
                    "task_index": index,
                    "question_id": result["question_id"],
                    "solved": result["solved"],
                    "tokens": result["total_generated_tokens"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    local = {"rank": rank, "seed": seed, "tasks": task_results}
    gathered: list[Any] | None = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        gathered.sort(key=lambda item: item["rank"])
        evidence = {
            "schema_version": 1,
            "fresh_run_started_unix": started,
            "elapsed_seconds": time.time() - started,
            "compute": {
                "backend": "kubernetes",
                "gpu_model": torch.cuda.get_device_name(0),
                "gpu_count": world,
            },
            "config": cfg,
            "summary": summarize(gathered, cfg),
            "seeds": gathered,
        }
        print("ORX_RESULT_JSON=" + json.dumps(evidence, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
