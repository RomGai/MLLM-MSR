"""Launch run_full_agents_pipeline_eval21.py across multiple GPUs via a shell wrapper.

Design goals:
- one process per GPU (task parallelism)
- deterministic user sharding by (--start-user-index, --max-users)
- optional shared DBs, with safer per-shard history DB by default
- unified final zip after ALL shards complete
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List
import time


@dataclass
class Shard:
    shard_id: int
    gpu_id: str
    start_user_index: int
    max_users: int


@dataclass
class DistEnv:
    world_size: int
    rank: int
    local_rank: int


def _read_int_env(keys: List[str], default: int) -> int:
    for k in keys:
        v = os.environ.get(k)
        if v is None or str(v).strip() == "":
            continue
        try:
            return int(v)
        except ValueError:
            continue
    return default


def _get_dist_env() -> DistEnv:
    world_size = _read_int_env(["WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"], 1)
    rank = _read_int_env(["RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"], 0)
    local_rank = _read_int_env(["LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"], rank)
    return DistEnv(world_size=world_size, rank=rank, local_rank=local_rank)


def _make_shards(total_users: int, gpu_ids: List[str]) -> List[Shard]:
    if total_users <= 0:
        raise ValueError("total_users must be > 0")
    if not gpu_ids:
        raise ValueError("gpu_ids must be non-empty")

    n = len(gpu_ids)
    base = total_users // n
    rem = total_users % n

    shards: List[Shard] = []
    start = 0
    for i, gid in enumerate(gpu_ids):
        size = base + (1 if i < rem else 0)
        if size <= 0:
            continue
        shards.append(Shard(shard_id=i, gpu_id=gid, start_user_index=start, max_users=size))
        start += size
    return shards


def _zip_dir(src_dir: Path, out_zip: Path) -> Path:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for child in src_dir.rglob("*"):
            if child.is_file():
                zf.write(child, arcname=str(child.relative_to(src_dir)))
    return out_zip


def _read_lines(p: Path, n: int = 30) -> str:
    if not p.exists():
        return ""
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-GPU launcher for eval21 pipeline")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--eval-script", default="./run_full_agents_pipeline_eval21.py")
    parser.add_argument("--sh-script", default="./run_eval21_shard.sh")

    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids")
    parser.add_argument("--total-users", type=int, required=True, help="Total users to process this run")
    parser.add_argument("--start-user-index", type=int, default=0, help="Global start offset")

    parser.add_argument("--eval-run-root", required=True, help="Shared run root for all shards")
    parser.add_argument(
        "--final-bundle-output",
        required=True,
        help="Final merged zip path (or directory, then <eval_run_root_name>.zip is used)",
    )

    parser.add_argument(
        "--shared-global-db-path",
        default="",
        help="Optional one global DB path shared across all shards (may hit sqlite write lock if cold-start)",
    )
    parser.add_argument(
        "--shared-history-db-path",
        default="",
        help="Optional one history DB path shared across all shards (NOT recommended; write contention risk)",
    )

    parser.add_argument(
        "--extra-args-json",
        default="[]",
        help="JSON array of extra args forwarded to eval script, e.g. '[\"--negative-sample-count\",\"99\"]'",
    )
    return parser


def _resolve_dist_gpu_id(dist: DistEnv, shard_gpu_id: str, gpu_ids: List[str]) -> str:
    """Choose GPU id passed to shard wrapper in distributed launch.

    - If launcher already masks each rank to a single GPU via CUDA_VISIBLE_DEVICES,
      pass "0" to stay inside that one visible device.
    - Otherwise prefer local-rank based mapping when possible.
    - Fallback to shard-assigned gpu id.
    """
    cvd = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if cvd:
        visible = [x.strip() for x in cvd.split(",") if x.strip()]
        if len(visible) == 1:
            return "0"

    if 0 <= dist.local_rank < len(gpu_ids):
        return gpu_ids[dist.local_rank]
    return shard_gpu_id


def main(args: argparse.Namespace) -> None:
    gpu_ids = [x.strip() for x in str(args.gpu_ids).split(",") if x.strip()]
    shards = _make_shards(int(args.total_users), gpu_ids)
    dist = _get_dist_env()

    eval_run_root = Path(args.eval_run_root).resolve()
    eval_run_root.mkdir(parents=True, exist_ok=True)

    final_bundle_output = Path(args.final_bundle_output).resolve()
    if final_bundle_output.exists() and final_bundle_output.is_dir():
        final_bundle_output = final_bundle_output / f"{eval_run_root.name}.zip"
    elif not final_bundle_output.exists() and final_bundle_output.suffix.lower() != ".zip":
        final_bundle_output.mkdir(parents=True, exist_ok=True)
        final_bundle_output = final_bundle_output / f"{eval_run_root.name}.zip"

    # validate JSON early
    extra_args = json.loads(str(args.extra_args_json))
    if not isinstance(extra_args, list):
        raise ValueError("extra-args-json must be a JSON array")

    def build_shard_cmd(shard: Shard, shard_root: Path, shard_bundle: Path) -> List[str]:
        shared_global_db_path = str(args.shared_global_db_path).strip()
        if shared_global_db_path:
            global_db = shared_global_db_path
        else:
            global_db = str(shard_root / "global_item_features.db")

        shared_history_db_path = str(args.shared_history_db_path).strip()
        if shared_history_db_path:
            history_db = shared_history_db_path
        else:
            history_db = str(shard_root / f"history_gpu{shard.gpu_id}.db")

        return [
            str(Path(args.sh_script).resolve()),
            str(Path(args.python_bin).resolve()),
            str(Path(args.eval_script).resolve()),
            shard.gpu_id,
            str(int(args.start_user_index) + shard.start_user_index),
            str(shard.max_users),
            str(shard_root),
            str(shard_bundle),
            global_db,
            history_db,
            json.dumps(extra_args, ensure_ascii=False),
        ]

    if dist.world_size > 1:
        if dist.rank >= len(shards):
            print(
                f"[dist] rank={dist.rank} world_size={dist.world_size} has no shard assigned "
                f"(shards={len(shards)}), exiting."
            )
            return

        shard = shards[dist.rank]
        shard_root = eval_run_root / f"shard_{shard.shard_id}"
        shard_root.mkdir(parents=True, exist_ok=True)
        shard_bundle = shard_root / "shard_bundle.zip"

        launch_gpu_id = _resolve_dist_gpu_id(dist, shard.gpu_id, gpu_ids)
        cmd = build_shard_cmd(
            Shard(
                shard_id=shard.shard_id,
                gpu_id=launch_gpu_id,
                start_user_index=shard.start_user_index,
                max_users=shard.max_users,
            ),
            shard_root,
            shard_bundle,
        )
        log_path = shard_root / f"launcher_rank{dist.rank}.log"
        err_path = shard_root / f"launcher_rank{dist.rank}.err"
        status_dir = eval_run_root / "_dist_status"
        status_dir.mkdir(parents=True, exist_ok=True)
        status_path = status_dir / f"rank_{dist.rank}.json"

        print(
            f"[dist-launch] rank={dist.rank}/{dist.world_size} local_rank={dist.local_rank} "
            f"-> shard={shard.shard_id} gpu={launch_gpu_id} start={cmd[4]} max={shard.max_users}"
        )
        with log_path.open("w", encoding="utf-8") as lf, err_path.open("w", encoding="utf-8") as ef:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=ef, env=os.environ.copy())
            ret = proc.wait()

        status_path.write_text(
            json.dumps(
                {
                    "rank": dist.rank,
                    "world_size": dist.world_size,
                    "ret": ret,
                    "shard_id": shard.shard_id,
                    "gpu_id": shard.gpu_id,
                    "launch_gpu_id": launch_gpu_id,
                    "log_path": str(log_path),
                    "err_path": str(err_path),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if dist.rank == 0:
            print("[dist] rank0 waiting for all rank status files...")
            timeout_s = 24 * 3600
            start_ts = time.time()
            expected = min(dist.world_size, len(shards))
            while True:
                present = list(status_dir.glob("rank_*.json"))
                if len(present) >= expected:
                    break
                if time.time() - start_ts > timeout_s:
                    raise RuntimeError("Timed out waiting for distributed shard completion")
                time.sleep(2)

            failed = []
            for i in range(expected):
                s = json.loads((status_dir / f"rank_{i}.json").read_text(encoding="utf-8"))
                if int(s.get("ret", 1)) != 0:
                    failed.append(s)

            if failed:
                print("\n[error] some distributed shards failed:")
                for item in failed:
                    print(
                        f"- rank={item['rank']} shard={item['shard_id']} gpu={item['gpu_id']} ret={item['ret']}"
                    )
                    print("  --- stderr tail ---")
                    print(_read_lines(Path(item["err_path"]), n=40))
                    print("  --- stdout tail ---")
                    print(_read_lines(Path(item["log_path"]), n=20))
                raise SystemExit(1)

            out_zip = _zip_dir(eval_run_root, final_bundle_output)
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "mode": "distributed",
                        "eval_run_root": str(eval_run_root),
                        "final_bundle_output": str(out_zip),
                        "shards": [
                            {
                                "shard_id": s.shard_id,
                                "gpu_id": s.gpu_id,
                                "start_user_index": int(args.start_user_index) + s.start_user_index,
                                "max_users": s.max_users,
                            }
                            for s in shards
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif ret != 0:
            raise SystemExit(ret)
        return

    procs = []
    for shard in shards:
        shard_root = eval_run_root / f"shard_{shard.shard_id}"
        shard_root.mkdir(parents=True, exist_ok=True)

        shard_bundle = shard_root / "shard_bundle.zip"

        log_path = shard_root / "launcher.log"
        err_path = shard_root / "launcher.err"
        lf = log_path.open("w", encoding="utf-8")
        ef = err_path.open("w", encoding="utf-8")

        cmd = build_shard_cmd(shard, shard_root, shard_bundle)

        print(f"[launch] shard={shard.shard_id} gpu={shard.gpu_id} start={cmd[4]} max={shard.max_users}")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=ef, env=os.environ.copy())
        procs.append((shard, proc, lf, ef, log_path, err_path))

    failed = []
    for shard, proc, lf, ef, log_path, err_path in procs:
        ret = proc.wait()
        lf.close()
        ef.close()
        if ret != 0:
            failed.append((shard, ret, log_path, err_path))

    if failed:
        print("\n[error] some shards failed:")
        for shard, ret, log_path, err_path in failed:
            print(f"- shard={shard.shard_id} gpu={shard.gpu_id} ret={ret}")
            print("  --- stderr tail ---")
            print(_read_lines(err_path, n=40))
            print("  --- stdout tail ---")
            print(_read_lines(log_path, n=20))
        raise SystemExit(1)

    out_zip = _zip_dir(eval_run_root, final_bundle_output)
    print(
        json.dumps(
            {
                "status": "ok",
                "eval_run_root": str(eval_run_root),
                "final_bundle_output": str(out_zip),
                "shards": [
                    {
                        "shard_id": s.shard_id,
                        "gpu_id": s.gpu_id,
                        "start_user_index": int(args.start_user_index) + s.start_user_index,
                        "max_users": s.max_users,
                    }
                    for s in shards
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main(build_argparser().parse_args())
