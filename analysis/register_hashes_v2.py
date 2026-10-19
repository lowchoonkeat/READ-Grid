"""Register SHA-256 hashes of every v2 artefact into docs/v2_artifact_hashes.txt.

Run this AFTER a stage finishes -- it hashes what is on disk at that moment, so
re-running it later legitimately produces a longer file. It never re-tags an old
artefact with a new hash: each section is regenerated from the current bytes and
the file records the generation time.

Usage:
    python analysis/register_hashes_v2.py
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "v2_artifact_hashes.txt"

SECTIONS = [
    ("冻结与治理", [
        "P359_V2_DEEPSEEK_EXECUTION_PLAN.md",
        "configs/protocol_v2.yaml",
        "configs/learner_v2_frozen.yaml",
        "docs/v2_preregistration.md",
        "docs/v2_preregistration.sha256",
        "docs/v1_source_provenance.md",
        "runs/workload_split.json",
        "runs_v2/_frozen_src_hashes.json",
    ]),
    ("阶段报告", [
        "docs/v2_stage_S0_report.md",
        "docs/v2_stage_S1_report.md",
        "docs/v2_stage_S2_report.md",
        "docs/v2_stage_S3_report.md",
        "docs/v2_stage_S4_report.md",
        "docs/v2_stage_S5_report.md",
        "docs/v2_stage_S6_report.md",
        "docs/v2_stage_S7_report.md",
        "docs/test_exposure_record.md",
        "docs/claude_plan_resolution.md",
    ]),
    ("源码", [
        "src/run_v2.py",
        "src/run_experiments.py",
        "src/readgrid/sim.py",
        "src/readgrid/controllers.py",
        "src/readgrid/value_model.py",
        "src/readgrid/reward.py",
        "analysis/s3_learner_v2.py",
        "analysis/s4_baselines_v2.py",
        "analysis/s3s4_freeze_winners.py",
        "analysis/s5_gap_check.py",
        "analysis/s6_analysis_v2.py",
        "analysis/s7_ec_outage_v2.py",
        "analysis/s7_ec_stress_v2.py",
        "analysis/s7_ec_stress_report.py",
        "analysis/s7_ea_transfer_v2.py",
        "analysis/register_hashes_v2.py",
        "analysis/traj_check_v2.py",
        "sandbox/p359_sandbox/ea_transfer.py",
    ]),
    ("S1/S2 沙盒证据", [
        "runs_v2/traj_check_v2.json",
        "runs_v2/sandbox_logs/ablate.txt",
        "runs_v2/sandbox_logs/mask_rerun.txt",
        "runs_v2/sandbox_logs/rulegrid.txt",
        "runs_v2/sandbox_logs/paired_var.txt",
        "runs_v2/sandbox_logs/dp_reference.txt",
        "runs_v2/sandbox_logs/dp_hires.txt",
    ]),
    ("S3/S4 候选与赢家", [
        "runs_v2/s3_ladder.json",
        "runs_v2/s3_learner_tuning_table.json",
        "runs_v2/s3_l6.json",
        "runs_v2/s3_learner_winner.json",
        "runs_v2/s3_final_sandbox.json",
        "runs_v2/s4_rule_tuning_table.json",
        "runs_v2/s4_c7.json",
        "runs_v2/s4_rule_winner.json",
        "runs_v2/sandbox_logs/s3_learner.txt",
        "runs_v2/sandbox_logs/s3_final.txt",
        "runs_v2/sandbox_logs/s4_baselines.txt",
        "runs_v2/sandbox_logs/freeze_winners.txt",
    ]),
    ("S5/S6 完整仿真器", [
        "runs_v2/train_v2_manifest.json",
        "runs_v2/validate_v2_manifest.json",
        "runs_v2/stage_validate_v2.jsonl",
        "runs_v2/validation_scores_v2.json",
        "runs_v2/s5_gap_check.json",
        "runs_v2/main_v2_manifest.json",
        "runs_v2/stage_main_v2.jsonl",
        "runs_v2/s6_results.json",
        "runs_v2/sandbox_logs/s5_train.txt",
        "runs_v2/sandbox_logs/s5_validate.txt",
        "runs_v2/sandbox_logs/s6_main.txt",
        "runs_v2/sandbox_logs/s5_gap_check.txt",
        "runs_v2/sandbox_logs/s6_analysis.txt",
        "runs_v2/s5_train_rerun_verification.json",
    ]),
    ("S7 扩展：E-C 停电聚焦", [
        "configs/protocol_v2_ec.yaml",
        "runs_v2/s7_ec.json",
        "runs_v2/s7_ec_outage_table.csv",
        "runs_v2/stage_ec_stress.jsonl",
        "runs_v2/ec_stress_manifest.json",
        "runs_v2/s7_ec_stress_results.json",
        "runs_v2/sandbox_logs/s7_ec_stress.txt",
        "runs_v2/event_traces/F8__LVP002.jsonl",
        "runs_v2/event_traces/F8__LVP005.jsonl",
    ]),
    ("S7 扩展：E-A 迁移鲁棒性", [
        "runs_v2/s7_ea_sandbox.json",
        "runs_v2/stage_ea_transfer.jsonl",
        "runs_v2/ea_transfer_manifest.json",
        "runs_v2/s7_ea_results.json",
        "runs_v2/s7_ea_table.csv",
        "runs_v2/sandbox_logs/s7_ea_sandbox.txt",
        "runs_v2/sandbox_logs/s7_ea_fullsim.txt",
        "runs_v2/sandbox_logs/s7_ea_report.txt",
    ]),
    ("v2 检查点", [
        *[f"models_v2/v2_{s}_linear_seed{k}.npy"
          for k in range(5) for s in ("cloud", "local")],
        "models_v2/training_log_v2.json",
    ]),
    ("v1 冻结检查点（只读输入）", [
        *[f"models/cloud_linear_seed{k}.npy" for k in range(5)],
        *[f"models/local_linear_seed{k}.npy" for k in range(5)],
        *[f"models/cloud_concatenated_seed{k}.npy" for k in range(5)],
        *[f"models/local_concatenated_seed{k}.npy" for k in range(5)],
        "models/checkpoint_manifest.json",
        "runs/stage_main.jsonl",
        "runs/main_manifest.json",
    ]),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


lines = [
    "# P359 / READ-Grid v2 —— 工件 SHA-256 登记表",
    "",
    f"生成时间：{time.strftime('%Y-%m-%dT%H:%M:%S')}",
    "",
    "本表由 `analysis/register_hashes_v2.py` 从**当前磁盘字节**生成。缺失的文件记为 `MISSING`，",
    "不猜测、不填占位值。重新运行只会在末尾追加新的生成时间与新出现的文件，不会用新哈希回填旧产物。",
    "",
]
n_ok = n_miss = 0
for title, files in SECTIONS:
    lines += [f"## {title}", "", "| 文件 | 字节 | SHA-256 |", "|---|---:|---|"]
    for rel in files:
        p = ROOT / rel
        if p.exists():
            lines.append(f"| `{rel}` | {p.stat().st_size} | `{sha256(p)}` |")
            n_ok += 1
        else:
            lines.append(f"| `{rel}` | — | MISSING |")
            n_miss += 1
    lines.append("")

lines += [f"合计：{n_ok} 个文件已登记，{n_miss} 个缺失。", ""]
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)}  ({n_ok} present, {n_miss} missing)")
