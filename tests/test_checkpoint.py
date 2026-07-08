"""Unit tests for eval/checkpoint.py —— 断点续跑核心逻辑。

覆盖：
- 原子追加写 + 加载已完成 case_id
- 旧 *_detail.json → *.jsonl 迁移（一次 + 二次 no-op）
- 损坏 JSON 行 warn + 跳过
- 重复 case_id → first-write wins
- --no-resume 触发 clear_checkpoint
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.checkpoint import (
    CheckpointWriter,
    clear_checkpoint,
    detail_path,
    iter_completed_cases,
    load_all_completed_cases,
    load_completed_case_ids,
    migrate_legacy_detail_if_needed,
)


def test_append_then_load(tmp_path: Path):
    """写两行 → 读出两条 case + 正确的 case_id set。"""
    p = tmp_path / "B1.jsonl"
    with CheckpointWriter(p) as w:
        w.append_case({"case_id": "c1", "hop_count": 2, "score": 0.5})
        w.append_case({"case_id": "c2", "hop_count": 2, "score": 0.7})

    assert p.exists()
    ids = load_completed_case_ids(tmp_path, "B1")
    assert ids == {"c1", "c2"}

    rows = load_all_completed_cases(tmp_path, "B1")
    assert len(rows) == 2
    assert rows[0]["case_id"] == "c1"
    assert rows[1]["case_id"] == "c2"
    assert rows[0]["score"] == 0.5


def test_migrate_legacy(tmp_path: Path):
    """老 detail.json 存在 → 自动迁移 → 二次迁移 no-op。"""
    detail = detail_path(tmp_path, "B1")
    detail.write_text(
        json.dumps(
            {
                "baseline_name": "B1",
                "case_results": [
                    {"case_id": "c1", "hop_count": 2, "score": 0.1},
                    {"case_id": "c2", "hop_count": 2, "score": 0.2},
                ],
            }
        ),
        encoding="utf-8",
    )
    # 首次迁移
    n = migrate_legacy_detail_if_needed(tmp_path, "B1")
    assert n == 2
    assert load_completed_case_ids(tmp_path, "B1") == {"c1", "c2"}

    # 二次迁移应 no-op
    n2 = migrate_legacy_detail_if_needed(tmp_path, "B1")
    assert n2 == 0


def test_migrate_empty_detail(tmp_path: Path):
    """detail.json 存在但 case_results 为空 → 迁移 0 条 + 警告。"""
    detail = detail_path(tmp_path, "B1")
    detail.write_text(
        json.dumps({"baseline_name": "B1", "case_results": []}),
        encoding="utf-8",
    )
    n = migrate_legacy_detail_if_needed(tmp_path, "B1")
    assert n == 0
    assert not (tmp_path / "B1.jsonl").exists()


def test_corrupt_line_skipped(tmp_path: Path):
    """损坏的 JSON 行 warn + 跳过，不抛异常。"""
    p = tmp_path / "B1.jsonl"
    p.write_text(
        '{"case_id":"c1","score":0.5}\n'
        "NOT_JSON_HERE\n"
        '{"case_id":"c2","score":0.7}\n',
        encoding="utf-8",
    )
    rows = list(iter_completed_cases(tmp_path, "B1"))
    assert [r["case_id"] for r in rows] == ["c1", "c2"]
    ids = load_completed_case_ids(tmp_path, "B1")
    assert ids == {"c1", "c2"}


def test_duplicate_case_id_first_wins(tmp_path: Path):
    """同 case_id 重复时，保留最早出现的（first-write wins）。"""
    p = tmp_path / "B1.jsonl"
    with CheckpointWriter(p) as w:
        w.append_case({"case_id": "c1", "score": 0.1})
        w.append_case({"case_id": "c1", "score": 0.2})  # 重复
        w.append_case({"case_id": "c2", "score": 0.3})

    rows = load_all_completed_cases(tmp_path, "B1")
    assert len(rows) == 2
    # c1 保留最早那条 score=0.1
    c1 = next(r for r in rows if r["case_id"] == "c1")
    assert c1["score"] == 0.1


def test_no_resume_clears(tmp_path: Path):
    """clear_checkpoint 删除 .jsonl，不动 detail.json。"""
    p = tmp_path / "B1.jsonl"
    detail = detail_path(tmp_path, "B1")
    p.write_text('{"case_id":"c1"}\n', encoding="utf-8")
    detail.write_text('{"baseline_name":"B1","case_results":[]}', encoding="utf-8")

    clear_checkpoint(tmp_path, "B1")
    assert not p.exists()
    assert detail.exists()  # detail.json 没动


def test_checkpoint_writer_requires_with(tmp_path: Path):
    """未 with 打开就 append 应该 RuntimeError。"""
    p = tmp_path / "B1.jsonl"
    w = CheckpointWriter(p)
    with pytest.raises(RuntimeError, match="未打开"):
        w.append_case({"case_id": "c1"})


def test_check_in_path(tmp_path: Path):
    """checkpoint_path 和 detail_path 返回符合约定。"""
    from eval.checkpoint import checkpoint_path as cp_fn
    p = cp_fn(tmp_path, "B1")
    assert p == tmp_path / "B1.jsonl"
    dp = detail_path(tmp_path, "B1")
    assert dp == tmp_path / "B1_detail.json"


def test_resume_load_empty_returns_empty_set(tmp_path: Path):
    """jsonl 不存在时 load_completed_case_ids 返回空 set，不抛错。"""
    ids = load_completed_case_ids(tmp_path, "B1_NaiveRAG")
    assert ids == set()
