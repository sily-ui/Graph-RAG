"""Eval checkpoint —— 断点续跑核心逻辑。

每跑完一个 case 立即以 JSONL 格式追加一行到 <output>/<baseline>.jsonl，
单行 write + flush + fsync 保证单行级别原子性，进程意外中断最多丢 1 条。

启动时扫描 .jsonl 加载已完成的 case_id 集合，跳过这些 case。
旧格式 * _detail.json 会在启动时自动迁移到 .jsonl。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def checkpoint_path(output_dir: Path, baseline_name: str) -> Path:
    """<output_dir>/<baseline_name>.jsonl"""
    return output_dir / f"{baseline_name}.jsonl"


def detail_path(output_dir: Path, baseline_name: str) -> Path:
    """<output_dir>/<baseline_name>_detail.json（兼容老输出，跑完后从 .jsonl 重新聚合写入）"""
    return output_dir / f"{baseline_name}_detail.json"


def load_completed_case_ids(
    output_dir: Path,
    baseline_name: str,
) -> set[str]:
    """返回已完成的 case_id 集合。

    损坏行 warn + 跳过，不抛异常。空 / 不存在文件返回空 set。
    """
    p = checkpoint_path(output_dir, baseline_name)
    done: set[str] = set()
    if not p.exists():
        return done
    try:
        with p.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    cid = d.get("case_id")
                    if cid:
                        done.add(cid)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[{baseline_name}] {p.name} 第 {lineno} 行 JSON 损坏，已跳过: "
                        f"{line[:80].strip()}"
                    )
    except OSError as e:
        logger.error(f"[{baseline_name}] 读 {p} 失败: {e}，按零进度处理")
        return set()
    return done


def migrate_legacy_detail_if_needed(
    output_dir: Path,
    baseline_name: str,
) -> int:
    """如果 .jsonl 不存在但 *_detail.json 存在，把 case_results 字段抽到 .jsonl。

    Returns
    -------
    int : 迁移的 case 数（0 表示无需迁移 / detail.json 解析失败 / 已经是新格式）
    """
    cp = checkpoint_path(output_dir, baseline_name)
    if cp.exists():
        # 已经有新格式，不迁移
        return 0
    dp = detail_path(output_dir, baseline_name)
    if not dp.exists():
        return 0
    try:
        with dp.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"[{baseline_name}] 迁移失败：读 {dp} 出错: {e}")
        return 0
    case_results = data.get("case_results") or []
    if not case_results:
        logger.warning(f"[{baseline_name}] {dp.name} 存在但 case_results 为空，跳过迁移")
        return 0
    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("a", encoding="utf-8") as f:
        for case in case_results:
            line = json.dumps(case, ensure_ascii=False, default=str)
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    logger.info(
        f"[{baseline_name}] 从 {dp.name} 迁移 {len(case_results)} 条 case 到 {cp.name}"
    )
    return len(case_results)


def clear_checkpoint(output_dir: Path, baseline_name: str) -> None:
    """--no-resume 时调用：删除 .jsonl。

    不动 *_detail.json —— 它会在本次 run 跑完后被新版本覆盖。
    """
    p = checkpoint_path(output_dir, baseline_name)
    if p.exists():
        p.unlink()
        logger.info(f"[{baseline_name}] --no-resume: 已清空 {p.name}")


def iter_completed_cases(
    output_dir: Path,
    baseline_name: str,
) -> Iterable[dict]:
    """从 .jsonl 逐行 yield case metrics dict。损坏行 warn + 跳过。"""
    p = checkpoint_path(output_dir, baseline_name)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    f"[{baseline_name}] {p.name} 第 {lineno} 行 JSON 损坏，已跳过: "
                    f"{line[:80].strip()}"
                )


def load_all_completed_cases(
    output_dir: Path,
    baseline_name: str,
) -> list[dict]:
    """从 .jsonl 加载全部 case metrics（用于最终聚合）。

    行为：
    - 同 case_id 重复时 first-write wins（保留最早出现的那条）
    - 损坏行 warn + 跳过
    """
    seen: dict[str, dict] = {}
    duplicates: set[str] = set()
    for case in iter_completed_cases(output_dir, baseline_name):
        cid = case.get("case_id")
        if cid is None:
            continue
        if cid in seen:
            duplicates.add(cid)
            continue
        seen[cid] = case
    if duplicates:
        logger.warning(
            f"[{baseline_name}] 发现重复 case_id（保留最早）: {sorted(duplicates)[:5]}"
            f"{' ...' if len(duplicates) > 5 else ''}"
        )
    return list(seen.values())


class CheckpointWriter:
    """单 baseline 的可复用 append 句柄。

    内部缓存 file descriptor（不每次 open），每次 append 走：
        write → flush → fsync
    单行级别原子：要么整行已落盘（reader 可见），要么整行不在（reader 不可见）。
    不会留下半行 JSONL，因为 write() 是单次系统调用。

    用法：
        with CheckpointWriter(checkpoint_path(output_dir, name)) as writer:
            writer.append_case(metrics)
    """

    def __init__(self, path: Path):
        self.path = path
        self._fp = None

    def __enter__(self) -> "CheckpointWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except OSError as e:
                logger.error(f"[CheckpointWriter] 关闭 {self.path} 失败: {e}")
            finally:
                self._fp = None

    def append_case(self, metrics: dict) -> None:
        """追加一条 case 的指标 dict。原子写：write + flush + fsync。"""
        if self._fp is None:
            raise RuntimeError(
                f"CheckpointWriter 未打开，请用 'with' 语句: {self.path}"
            )
        line = json.dumps(metrics, ensure_ascii=False, default=str) + "\n"
        self._fp.write(line)
        self._fp.flush()
        os.fsync(self._fp.fileno())
