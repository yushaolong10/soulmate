#!/usr/bin/env python3
"""并行执行 dpo_scripts/ 下所有 dpo_data*.py 脚本。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "dpo_scripts"
DEFAULT_LOG_DIR = ROOT / "logs" / "dpo_runs"


@dataclass
class RunResult:
    name: str
    returncode: int
    log_path: Path
    elapsed_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="并行执行 dpo_scripts/dpo_data*.py，并为每个脚本写独立日志。"
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="用于执行脚本的 Python 解释器，默认使用当前解释器。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(6, max(1, (os.cpu_count() or 4) // 2)),
        help="并行度，默认取 CPU 数的一半，最高默认 6。",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(DEFAULT_LOG_DIR),
        help="日志输出目录，默认 logs/dpo_runs。",
    )
    parser.add_argument(
        "--pattern",
        default="dpo_data*.py",
        help="脚本匹配模式，默认 dpo_data*.py。",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="只运行指定脚本名，可写多个，例如 --only dpo_data_time.py dpo_data_emoji.py",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="排除指定脚本名，可写多个。",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="任一脚本失败后停止继续提交新任务。",
    )
    return parser.parse_args()


def discover_scripts(
    scripts_dir: Path, pattern: str, only: Sequence[str], exclude: Sequence[str]
) -> List[Path]:
    scripts = sorted(p for p in scripts_dir.glob(pattern) if p.is_file())
    only_set = set(only)
    exclude_set = set(exclude)
    if only_set:
        scripts = [p for p in scripts if p.name in only_set]
    if exclude_set:
        scripts = [p for p in scripts if p.name not in exclude_set]
    return scripts


def run_one(script: Path, python_bin: str, log_dir: Path) -> RunResult:
    log_path = log_dir / f"{script.stem}.log"
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            [python_bin, str(script)],
            cwd=ROOT,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - start
    return RunResult(
        name=script.name,
        returncode=proc.returncode,
        log_path=log_path,
        elapsed_sec=elapsed,
    )


def submit_initial(
    executor: ThreadPoolExecutor,
    pending_scripts: List[Path],
    python_bin: str,
    log_dir: Path,
    max_workers: int,
):
    futures = {}
    while pending_scripts and len(futures) < max_workers:
        script = pending_scripts.pop(0)
        future = executor.submit(run_one, script, python_bin, log_dir)
        futures[future] = script
    return futures


def print_result(result: RunResult) -> None:
    status = "OK" if result.returncode == 0 else "FAIL"
    print(
        f"[{status}] {result.name}  "
        f"code={result.returncode}  "
        f"time={result.elapsed_sec:.1f}s  "
        f"log={result.log_path}"
    )


def print_summary(results: Iterable[RunResult]) -> int:
    results = list(results)
    failures = [r for r in results if r.returncode != 0]
    print("\n=== Summary ===")
    print(f"total={len(results)} success={len(results) - len(failures)} failed={len(failures)}")
    if failures:
        print("failed scripts:")
        for result in sorted(failures, key=lambda r: r.name):
            print(f"  - {result.name} -> {result.log_path}")
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    log_dir = Path(args.logs_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    scripts = discover_scripts(
        SCRIPTS_DIR,
        args.pattern,
        only=args.only,
        exclude=args.exclude,
    )
    if not scripts:
        print("没有找到可执行的 dpo_data 脚本。")
        return 1

    print(f"python: {args.python}")
    print(f"scripts: {len(scripts)}")
    print(f"max_workers: {args.max_workers}")
    print(f"logs_dir: {log_dir}")
    print("targets:")
    for script in scripts:
        print(f"  - {script.name}")
    print()

    results: List[RunResult] = []
    pending_scripts = list(scripts)
    stop_submitting = False

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = submit_initial(
            executor,
            pending_scripts,
            args.python,
            log_dir,
            args.max_workers,
        )

        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                script = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover
                    result = RunResult(
                        name=script.name,
                        returncode=1,
                        log_path=log_dir / f"{script.stem}.log",
                        elapsed_sec=0.0,
                    )
                    with open(result.log_path, "a", encoding="utf-8") as log_f:
                        log_f.write(f"\n[runner exception] {exc}\n")

                results.append(result)
                print_result(result)

                if args.fail_fast and result.returncode != 0:
                    stop_submitting = True

            if stop_submitting:
                continue

            while pending_scripts and len(futures) < args.max_workers:
                next_script = pending_scripts.pop(0)
                future = executor.submit(run_one, next_script, args.python, log_dir)
                futures[future] = next_script

    return print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
