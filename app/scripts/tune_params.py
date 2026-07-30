#!/usr/bin/env python3
"""
Tune localization parameter files per model.

The script generates candidate YAML files, runs `test_algs.launch` for each
candidate/bag/repeat, recomputes metrics with `offline_evaluate.py`, and writes
a per-model ranking plus the best YAML found for each mode.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


APP_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = APP_DIR.parent
PARAMS_DIR = APP_DIR / "params"
BAGS_DIR = APP_DIR / "bags"
RESULTS_DIR = APP_DIR / "results"
TUNING_RESULTS_DIR = REPO_DIR / "tuning_results"
DEFAULT_BASE_PARAM = PARAMS_DIR / "amhmcl_medium.yaml"

DEFAULT_MODES = ["MCL", "MHMCL", "AMCL", "MHAMCL", "3MCL"]
KNOWN_MODES = [
    "AMHAMCL",
    "AMHMCL",
    "MHAMCL",
    "MHMCL",
    "AMCL",
    "3MCL",
    "MCL",
]

PHYSICAL_KEYS = {
    "alpha1",
    "alpha2",
    "alpha3",
    "alpha4",
    "sigma_hit",
    "z_hit",
    "z_rand",
    "max_range",
    "step",
}

SENSOR_PROFILES = {
    "fast": {"sigma_hit": 0.12, "z_hit": 0.80, "z_rand": 0.20, "step": 2},
    "balanced": {"sigma_hit": 0.08, "z_hit": 0.85, "z_rand": 0.15, "step": 2},
    "accurate": {"sigma_hit": 0.06, "z_hit": 0.90, "z_rand": 0.10, "step": 1},
}

MOTION_PROFILES = {
    "low": {"alpha1": 0.08, "alpha2": 0.09, "alpha3": 0.10, "alpha4": 0.03},
    "medium": {"alpha1": 0.16, "alpha2": 0.175, "alpha3": 0.18, "alpha4": 0.05},
    "high": {"alpha1": 0.18, "alpha2": 0.18, "alpha3": 0.20, "alpha4": 0.06},
}

KLD_PROFILES = {
    "loose": {
        "kld_epsilon": 0.05,
        "kld_z": 1.96,
        "kld_bin_size_xy": 0.10,
        "kld_bin_size_theta": 0.17,
        "alpha_slow": 0.0001,
        "alpha_fast": 0.05,
        "min_ratio": 0.10,
        "max_mult": 2.0,
    },
    "balanced": {
        "kld_epsilon": 0.02,
        "kld_z": 2.326,
        "kld_bin_size_xy": 0.05,
        "kld_bin_size_theta": 0.087,
        "alpha_slow": 0.0005,
        "alpha_fast": 0.08,
        "min_ratio": 0.20,
        "max_mult": 3.0,
    },
    "robust": {
        "kld_epsilon": 0.01,
        "kld_z": 2.326,
        "kld_bin_size_xy": 0.05,
        "kld_bin_size_theta": 0.087,
        "alpha_slow": 0.0001,
        "alpha_fast": 0.03,
        "min_ratio": 0.25,
        "max_mult": 4.0,
    },
}

META_PROFILES = {
    "small_walk": {
        "alpha5": 0.08,
        "alpha6": 0.08,
        "alpha7": 0.08,
        "alpha8": 0.08,
        "random_steps": 20,
        "meta_lambda": 0.65,
    },
    "medium_walk": {
        "alpha5": 0.18,
        "alpha6": 0.18,
        "alpha7": 0.18,
        "alpha8": 0.18,
        "random_steps": 40,
        "meta_lambda": 0.72,
    },
    "wide_walk": {
        "alpha5": 0.30,
        "alpha6": 0.20,
        "alpha7": 0.30,
        "alpha8": 0.30,
        "random_steps": 60,
        "meta_lambda": 0.90,
    },
}


@dataclass
class Candidate:
    mode: str
    candidate_id: str
    labels: Dict[str, str]
    overrides: Dict[str, Any]
    params: Dict[str, Any]
    params_path: Path
    result_dir: Path


def strip_inline_comment(line: str) -> str:
    quote = None
    escaped = False
    out = []

    for char in line:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            out.append(char)
            escaped = True
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            out.append(char)
            continue
        if char == "#" and quote is None:
            break
        out.append(char)

    return "".join(out).strip()


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""

    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_flat_yaml(path: Path) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    with path.open() as f:
        for raw_line in f:
            line = strip_inline_comment(raw_line)
            if not line or ":" not in line or line.startswith("-"):
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if not key or " " in key:
                continue
            params[key] = parse_scalar(value)
    return params


def format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value))


def dump_flat_yaml(params: Dict[str, Any], path: Path, mode: str, labels: Dict[str, str]) -> None:
    groups = [
        (
            "General",
            ["initialized", "use_sim_time", "init_particles", "min_particles", "max_particles", "headless"],
        ),
        ("Motion model", ["alpha1", "alpha2", "alpha3", "alpha4"]),
        ("Meta / random walk", ["alpha5", "alpha6", "alpha7", "alpha8", "random_steps", "meta_lambda"]),
        (
            "KLD / AMCL",
            [
                "kld_epsilon",
                "kld_z",
                "kld_bin_size_xy",
                "kld_bin_size_theta",
                "alpha_slow",
                "alpha_fast",
            ],
        ),
        ("Sensor model", ["sigma_hit", "z_hit", "z_rand", "max_range", "step"]),
        ("Topics", ["odom_topic", "scan_topic", "initial_pose_topic"]),
    ]

    written = set()
    lines = [
        "# Auto-generated by app/scripts/tune_params.py",
        f"# Mode: {mode}",
        "# Labels: " + json.dumps(labels, sort_keys=True),
        "",
    ]

    for title, keys in groups:
        present = [key for key in keys if key in params]
        if not present:
            continue
        lines.append(f"# {title}")
        for key in present:
            lines.append(f"{key}: {format_scalar(params[key])}")
            written.add(key)
        lines.append("")

    extra = sorted(key for key in params if key not in written)
    if extra:
        lines.append("# Extra")
        for key in extra:
            lines.append(f"{key}: {format_scalar(params[key])}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def as_int(value: float) -> int:
    return max(1, int(round(value)))


def is_adaptive_mode(mode: str) -> bool:
    return "AMCL" in mode


def is_meta_mode(mode: str) -> bool:
    return "3" in mode


def candidate_digest(overrides: Dict[str, Any]) -> str:
    payload = json.dumps(overrides, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(payload).hexdigest()[:8]


def profile_names(preset: str, profile_type: str) -> List[str]:
    if profile_type == "sensor":
        return ["balanced"] if preset == "tiny" else ["balanced", "accurate"]
    if profile_type == "motion":
        return ["medium"] if preset == "tiny" else ["medium", "high"]
    if profile_type == "kld":
        if preset == "tiny":
            return ["balanced"]
        if preset == "quick":
            return ["loose", "balanced", "robust"]
        return ["loose", "balanced", "robust"]
    if profile_type == "meta":
        return ["medium_walk"] if preset == "tiny" else ["small_walk", "medium_walk", "wide_walk"]
    raise ValueError(profile_type)


def particle_counts_for(mode: str, preset: str) -> List[int]:
    if preset == "tiny":
        return [500, 750]
    if preset == "quick":
        if is_adaptive_mode(mode):
            return [300, 500, 750]
        return [300, 500, 750, 1000]
    if is_adaptive_mode(mode):
        return [150, 300, 500, 750, 1000, 1500]
    return [100, 300, 500, 750, 1000, 1500]


def limit_candidates(candidates: List[Tuple[Dict[str, str], Dict[str, Any]]], max_count: Optional[int]) -> List[Tuple[Dict[str, str], Dict[str, Any]]]:
    if max_count is None or max_count <= 0 or len(candidates) <= max_count:
        return candidates

    # Keep an even spread through the generated grid instead of taking only the
    # first values, which would bias toward small particle counts.
    if max_count == 1:
        return [candidates[len(candidates) // 2]]

    selected = []
    last_index = len(candidates) - 1
    for i in range(max_count):
        idx = int(round(i * last_index / (max_count - 1)))
        selected.append(candidates[idx])
    return selected


def build_candidate_specs(
    mode: str,
    preset: str,
    fair_physical: bool,
    max_candidates_per_mode: Optional[int],
) -> List[Tuple[Dict[str, str], Dict[str, Any]]]:
    particles = particle_counts_for(mode, preset)
    sensor_names = ["base"] if fair_physical else profile_names(preset, "sensor")
    motion_names = ["base"] if fair_physical else profile_names(preset, "motion")

    specs: List[Tuple[Dict[str, str], Dict[str, Any]]] = []

    if is_adaptive_mode(mode):
        kld_names = profile_names(preset, "kld")
        for n, sensor_name, motion_name, kld_name in itertools.product(
            particles, sensor_names, motion_names, kld_names
        ):
            overrides: Dict[str, Any] = {
                "init_particles": n,
                "headless": True,
            }
            labels = {
                "particles": str(n),
                "sensor": sensor_name,
                "motion": motion_name,
                "kld": kld_name,
            }
            if sensor_name != "base":
                overrides.update(SENSOR_PROFILES[sensor_name])
            if motion_name != "base":
                overrides.update(MOTION_PROFILES[motion_name])

            kld = dict(KLD_PROFILES[kld_name])
            min_ratio = float(kld.pop("min_ratio"))
            max_mult = float(kld.pop("max_mult"))
            overrides.update(kld)
            overrides["min_particles"] = as_int(n * min_ratio)
            overrides["max_particles"] = as_int(n * max_mult)
            specs.append((labels, overrides))

    elif is_meta_mode(mode):
        meta_names = profile_names(preset, "meta")
        for n, sensor_name, motion_name, meta_name in itertools.product(
            particles, sensor_names, motion_names, meta_names
        ):
            overrides = {
                "init_particles": n,
                "min_particles": as_int(n * 0.10),
                "max_particles": as_int(n * 2.0),
                "headless": True,
            }
            labels = {
                "particles": str(n),
                "sensor": sensor_name,
                "motion": motion_name,
                "meta": meta_name,
            }
            if sensor_name != "base":
                overrides.update(SENSOR_PROFILES[sensor_name])
            if motion_name != "base":
                overrides.update(MOTION_PROFILES[motion_name])
            overrides.update(META_PROFILES[meta_name])
            specs.append((labels, overrides))

    else:
        for n, sensor_name, motion_name in itertools.product(particles, sensor_names, motion_names):
            overrides = {
                "init_particles": n,
                "min_particles": as_int(n * 0.10),
                "max_particles": as_int(n * 2.0),
                "headless": True,
            }
            labels = {
                "particles": str(n),
                "sensor": sensor_name,
                "motion": motion_name,
            }
            if sensor_name != "base":
                overrides.update(SENSOR_PROFILES[sensor_name])
            if motion_name != "base":
                overrides.update(MOTION_PROFILES[motion_name])
            specs.append((labels, overrides))

    return limit_candidates(specs, max_candidates_per_mode)


def generate_candidates(args: argparse.Namespace, run_dir: Path) -> List[Candidate]:
    base_params = load_flat_yaml(args.base_param)
    candidates: List[Candidate] = []

    for mode in args.modes:
        specs = build_candidate_specs(
            mode,
            args.preset,
            args.fair_physical,
            args.max_candidates_per_mode,
        )
        for idx, (labels, overrides) in enumerate(specs, start=1):
            params = dict(base_params)
            params.update(overrides)
            digest = candidate_digest(overrides)
            candidate_id = f"{mode.lower()}_{idx:03d}_{digest}"
            params_path = run_dir / "params" / mode / f"{candidate_id}.yaml"
            result_dir = run_dir / "runs" / mode / candidate_id
            dump_flat_yaml(params, params_path, mode, labels)
            result_dir.mkdir(parents=True, exist_ok=True)

            meta = {
                "mode": mode,
                "candidate_id": candidate_id,
                "labels": labels,
                "overrides": overrides,
                "params_path": str(params_path),
                "particle_cost": int(params.get("init_particles", 0)),
            }
            (result_dir / "candidate.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            candidates.append(
                Candidate(
                    mode=mode,
                    candidate_id=candidate_id,
                    labels=labels,
                    overrides=overrides,
                    params=params,
                    params_path=params_path,
                    result_dir=result_dir,
                )
            )

    return candidates


def resolve_bag(path_or_name: str) -> List[Path]:
    path = Path(path_or_name).expanduser()
    if not path.exists():
        path = BAGS_DIR / path_or_name
    if path.is_dir():
        return sorted(path.glob("*.bag"))
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Bag not found: {path_or_name}")


def resolve_bags(items: Optional[List[str]]) -> List[Path]:
    if not items:
        default = BAGS_DIR / "explore_bin.bag"
        if default.exists():
            return [default]
        return sorted(BAGS_DIR.glob("*.bag"))[:1]

    bags: List[Path] = []
    for item in items:
        bags.extend(resolve_bag(item))
    bags = sorted(set(path.resolve() for path in bags))
    if not bags:
        raise FileNotFoundError("No .bag files found.")
    return bags


def command_succeeds(cmd: List[str], timeout: float = 5.0) -> bool:
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class Roscore:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "Roscore":
        if not self.enabled:
            return self
        if command_succeeds(["rostopic", "list"], timeout=5.0):
            return self

        print("Starting roscore...")
        self.proc = subprocess.Popen(
            ["roscore"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        for _ in range(60):
            if command_succeeds(["rostopic", "list"], timeout=2.0):
                print("roscore is ready.")
                return self
            time.sleep(1.0)
        raise RuntimeError("roscore did not become ready within 60 seconds")

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass


def terminate_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def run_one_candidate(
    candidate: Candidate,
    bag: Path,
    repeat: int,
    args: argparse.Namespace,
) -> int:
    result_name = f"{bag.stem}_{candidate.mode}_{candidate.candidate_id}_run{repeat}"
    env = os.environ.copy()
    env["BAG_FILE"] = str(bag)
    env.setdefault("ROS_MASTER_URI", "http://localhost:11311")
    env.setdefault("ROS_HOSTNAME", "localhost")

    cmd = [
        "roslaunch",
        "mcmh_localization",
        "test_algs.launch",
        f"mode:={candidate.mode}",
        f"result_name:={result_name}",
        f"robot_name:={args.robot_name}",
        f"param_file:={candidate.params_path}",
        f"results_dir:={candidate.result_dir}",
    ]

    print(
        f"[run] mode={candidate.mode} candidate={candidate.candidate_id} "
        f"bag={bag.name} repeat={repeat}/{args.repeats}"
    )
    if args.dry_run:
        print("      " + " ".join(str(part) for part in cmd))
        return 0

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_DIR),
        env=env,
        stdout=None if args.verbose else subprocess.DEVNULL,
        stderr=None if args.verbose else subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    timed_out = False
    try:
        return_code = proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_group(proc)
        return_code = 124

    if args.kill_rosnodes and not args.dry_run:
        subprocess.run(
            ["rosnode", "kill", "-a"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        time.sleep(args.cooldown)

    if timed_out:
        print(f"[timeout] {candidate.candidate_id} | {bag.name} | run {repeat}")
    elif return_code != 0:
        print(f"[warn] roslaunch returned {return_code} for {candidate.candidate_id}")

    return return_code


def run_experiments(candidates: List[Candidate], bags: List[Path], args: argparse.Namespace) -> None:
    total = len(candidates) * len(bags) * args.repeats
    print(
        f"Running {total} experiments "
        f"({len(candidates)} candidates x {len(bags)} bags x {args.repeats} repeats)."
    )

    with Roscore(enabled=not args.no_roscore and not args.dry_run):
        done = 0
        for candidate in candidates:
            for bag in bags:
                for repeat in range(1, args.repeats + 1):
                    run_one_candidate(candidate, bag, repeat, args)
                    done += 1
                    print(f"[progress] {done}/{total}")


def discover_pose_result_dirs(root: Path) -> List[Path]:
    result_dirs = []
    for current_dir, dirnames, filenames in os.walk(root):
        dirnames[:] = [dirname for dirname in dirnames if dirname != "plots"]
        if any(name.startswith("poses_") and name.endswith(".txt") for name in filenames):
            result_dirs.append(Path(current_dir))
    return sorted(result_dirs)


def run_offline_evaluation(run_dir: Path, args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    sys.path.insert(0, str(APP_DIR / "scripts"))
    import offline_evaluate

    print("Recomputing offline metrics...")
    result_dirs = discover_pose_result_dirs(run_dir)
    if not result_dirs:
        print(f"No pose files found under: {run_dir}")
        return
    for result_dir in result_dirs:
        offline_evaluate.process_results_dir(str(result_dir))


def safe_float(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def std(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def read_summary(summary_path: Path) -> List[Dict[str, str]]:
    if not summary_path.exists():
        return []
    with summary_path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_candidate_meta(result_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = result_dir / "candidate.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def infer_mode_from_file(filename: str) -> Optional[str]:
    for mode in KNOWN_MODES:
        if f"_{mode}_" in filename:
            return mode
    return None


def collect_candidate_result_dirs(run_dir: Path) -> List[Path]:
    dirs = []
    for summary_path in run_dir.rglob("summary_results.txt"):
        if "plots" in summary_path.parts:
            continue
        dirs.append(summary_path.parent)
    return sorted(set(dirs))


def summarize_result_dir(result_dir: Path, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    rows = read_summary(result_dir / "summary_results.txt")
    if not rows:
        return None

    meta = load_candidate_meta(result_dir) or {}
    mode = meta.get("mode")
    if mode is None:
        mode = infer_mode_from_file(rows[0].get("file", "")) or "unknown"

    rmse_pos = [v for row in rows if (v := safe_float(row.get("rmse_pos", ""))) is not None]
    rmse_yaw = [v for row in rows if (v := safe_float(row.get("rmse_yaw_rad", ""))) is not None]
    success = [v for row in rows if (v := safe_float(row.get("success", ""))) is not None]
    spl = [v for row in rows if (v := safe_float(row.get("spl", ""))) is not None]
    recall_t1 = [v for row in rows if (v := safe_float(row.get("recall_t1", ""))) is not None]
    recall_t2 = [v for row in rows if (v := safe_float(row.get("recall_t2", ""))) is not None]
    recall_t3 = [v for row in rows if (v := safe_float(row.get("recall_t3", ""))) is not None]
    failure_events = [v for row in rows if (v := safe_float(row.get("failure_events", ""))) is not None]
    failure_rate = [
        v
        for row in rows
        if (v := safe_float(row.get("failure_rate_events_per_km", ""))) is not None
    ]

    if not rmse_pos:
        return None

    particle_cost = meta.get("particle_cost")
    if particle_cost is None:
        particle_cost = 0
        labels = meta.get("labels", {})
        if isinstance(labels, dict):
            particle_cost = int(labels.get("particles", 0) or 0)

    stats = {
        "mode": mode,
        "candidate_id": meta.get("candidate_id", result_dir.name),
        "result_dir": str(result_dir),
        "params_path": meta.get("params_path", ""),
        "labels": meta.get("labels", {}),
        "overrides": meta.get("overrides", {}),
        "runs": len(rows),
        "rmse_pos_mean": mean(rmse_pos),
        "rmse_pos_std": std(rmse_pos),
        "rmse_yaw_mean": mean(rmse_yaw) or 0.0,
        "rmse_yaw_std": std(rmse_yaw) or 0.0,
        "success_rate": mean(success) or 0.0,
        "spl_mean": mean(spl) or 0.0,
        "recall_t1_mean": mean(recall_t1) or 0.0,
        "recall_t2_mean": mean(recall_t2) or 0.0,
        "recall_t3_mean": mean(recall_t3) or 0.0,
        "failure_events_mean": mean(failure_events) or 0.0,
        "failure_rate_mean": mean(failure_rate) or 0.0,
        "particle_cost": int(particle_cost or 0),
    }
    stats["score"] = (
        stats["rmse_pos_mean"]
        + args.yaw_weight * stats["rmse_yaw_mean"]
        + args.failure_penalty * (1.0 - stats["success_rate"])
        + args.failure_event_weight * stats["failure_events_mean"]
        + args.failure_rate_weight * stats["failure_rate_mean"]
        + args.particle_weight * stats["particle_cost"]
    )
    return stats


def write_ranking(run_dir: Path, rows: List[Dict[str, Any]]) -> None:
    ranking_path = run_dir / "ranking.csv"
    fields = [
        "rank",
        "mode",
        "candidate_id",
        "score",
        "runs",
        "rmse_pos_mean",
        "rmse_pos_std",
        "rmse_yaw_mean",
        "rmse_yaw_std",
        "success_rate",
        "spl_mean",
        "recall_t1_mean",
        "recall_t2_mean",
        "recall_t3_mean",
        "failure_events_mean",
        "failure_rate_mean",
        "particle_cost",
        "params_path",
        "result_dir",
        "labels",
        "overrides",
    ]
    with ranking_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = rank
            out["labels"] = json.dumps(out.get("labels", {}), sort_keys=True)
            out["overrides"] = json.dumps(out.get("overrides", {}), sort_keys=True)
            writer.writerow({field: out.get(field, "") for field in fields})
    print(f"Ranking saved: {ranking_path}")


def copy_best_params(run_dir: Path, rows: List[Dict[str, Any]]) -> None:
    best_dir = run_dir / "best_params"
    best_dir.mkdir(parents=True, exist_ok=True)
    seen = set()

    for row in rows:
        mode = row["mode"]
        if mode in seen:
            continue
        seen.add(mode)
        params_path = row.get("params_path")
        if params_path and Path(params_path).exists():
            dest = best_dir / f"{mode}.yaml"
            shutil.copyfile(params_path, dest)
            print(f"Best {mode}: {dest} | score={row['score']:.4f}")


def rank_results(run_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    summaries = []
    for result_dir in collect_candidate_result_dirs(run_dir):
        stats = summarize_result_dir(result_dir, args)
        if stats is not None:
            summaries.append(stats)

    summaries.sort(key=lambda row: (row["mode"], row["score"]))

    ranked = []
    for mode in sorted(set(row["mode"] for row in summaries)):
        mode_rows = [row for row in summaries if row["mode"] == mode]
        ranked.extend(mode_rows)

    if ranked:
        write_ranking(run_dir, ranked)
        copy_best_params(run_dir, ranked)
    else:
        print(f"No summaries found under: {run_dir}")
    return ranked


def print_top(rows: List[Dict[str, Any]], top_n: int) -> None:
    if not rows:
        return
    print("\nTop candidates per mode:")
    for mode in sorted(set(row["mode"] for row in rows)):
        print(f"\n[{mode}]")
        mode_rows = [row for row in rows if row["mode"] == mode][:top_n]
        for idx, row in enumerate(mode_rows, start=1):
            labels = ", ".join(f"{k}={v}" for k, v in sorted(row.get("labels", {}).items()))
            print(
                f"{idx:>2}. score={row['score']:.4f} "
                f"rmse={row['rmse_pos_mean']:.4f}+/-{row['rmse_pos_std']:.4f} "
                f"yaw={row['rmse_yaw_mean']:.4f} "
                f"success={row['success_rate']:.2f} "
                f"candidate={row['candidate_id']} "
                f"{labels}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune MCMH localization params per model and rank the results."
    )
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--bags", nargs="+", help="Bag files, bag names, or directories. Default: explore_bin.bag.")
    parser.add_argument("--base-param", type=Path, default=DEFAULT_BASE_PARAM)
    parser.add_argument("--results-root", type=Path, default=TUNING_RESULTS_DIR)
    parser.add_argument("--run-name", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--preset", choices=["tiny", "quick", "full"], default="quick")
    parser.add_argument("--max-candidates-per-mode", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=100)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--robot-name", default=f"turtlebot3_{os.environ.get('TURTLEBOT3_MODEL', 'waffle')}")
    parser.add_argument("--fair-physical", action="store_true", help="Keep physical robot/sensor params fixed across modes.")
    parser.add_argument("--rank-only", type=Path, help="Only rank an existing tuning directory.")
    parser.add_argument("--dry-run", action="store_true", help="Generate candidates and print roslaunch commands without running.")
    parser.add_argument("--no-roscore", action="store_true", help="Do not start roscore automatically.")
    parser.add_argument("--no-kill-rosnodes", dest="kill_rosnodes", action="store_false")
    parser.add_argument("--verbose", action="store_true", help="Show roslaunch output.")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--yaw-weight", type=float, default=0.2)
    parser.add_argument("--failure-penalty", type=float, default=1.0)
    parser.add_argument("--failure-event-weight", type=float, default=0.05)
    parser.add_argument("--failure-rate-weight", type=float, default=0.0)
    parser.add_argument("--particle-weight", type=float, default=0.00005)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.base_param = args.base_param.resolve()

    if args.rank_only:
        run_dir = args.rank_only.resolve()
        rows = rank_results(run_dir, args)
        print_top(rows, args.top)
        return 0 if rows else 1

    bags = resolve_bags(args.bags)
    run_dir = (args.results_root / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates = generate_candidates(args, run_dir)
    manifest = {
        "run_dir": str(run_dir),
        "base_param": str(args.base_param),
        "modes": args.modes,
        "bags": [str(path) for path in bags],
        "preset": args.preset,
        "repeats": args.repeats,
        "timeout": args.timeout,
        "fair_physical": args.fair_physical,
        "candidate_count": len(candidates),
        "scoring": {
            "yaw_weight": args.yaw_weight,
            "failure_penalty": args.failure_penalty,
            "failure_event_weight": args.failure_event_weight,
            "failure_rate_weight": args.failure_rate_weight,
            "particle_weight": args.particle_weight,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Tuning run directory: {run_dir}")
    print(f"Candidates: {len(candidates)}")
    print("Bags: " + ", ".join(path.name for path in bags))

    run_experiments(candidates, bags, args)
    run_offline_evaluation(run_dir, args)
    rows = rank_results(run_dir, args)
    print_top(rows, args.top)

    if args.dry_run:
        print("\nDry run complete. Remove --dry-run to execute the experiments.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
