"""扩展图库 —— 为 4 跳评估合成更多节点和路径。

现有图库只有 2/3 跳路径，0 条 4 跳路径。给 evaluator 跑 4 跳测试前
需要先扩充图库。本脚本通过 Neo4j 直接插入合成节点和边，复用现有
节点标签和 RELATES_TO + name 属性模式。
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from random import Random

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv('/root/.trae-cn/memory/../..Graph-RAG/.env') if False else load_dotenv('/root/Graph-RAG/.env')

# 合成节点定义
SYNTHETIC_COMPONENTS = [
    "machine-1-4", "machine-1-5", "machine-2-1", "machine-2-2",
    "machine-2-3", "machine-3-1", "machine-3-2",
    "kubelet", "kube-proxy", "etcd", "coredns",
    "frontend-pod", "backend-pod", "db-pod", "cache-pod",
]

SYNTHETIC_SYMPTOMS = [
    "memory_usage", "disk_io", "network_in", "network_out",
    "io_wait", "load_average", "packet_loss", "dns_error",
    "auth_failure", "oom_killed", "throttled_cpus", "fs_full",
    "connection_reset", "tls_handshake_fail", "gc_pause",
]

SYNTHETIC_CAUSES = [
    "memory_leak", "disk_pressure", "noisy_neighbor", "config_drift",
    "oom_event", "kernel_panic", "dns_timeout", "certificate_expired",
    "load_spike", "connection_pool_exhausted", "gc_pressure",
    "tcp_backlog", "cgroup_limit", "disk_quota_exceeded",
    "scheduling_latency", "network_saturation",
]

SYNTHETIC_SOLUTIONS = [
    "runbook-azure/memory-leak", "runbook-azure/disk-cleanup",
    "runbook-azure/noisy-neighbor", "runbook-azure/config-audit",
    "runbook-azure/oom-investigation", "runbook-azure/kernel-debug",
    "runbook-azure/dns-troubleshoot", "runbook-azure/renew-cert",
    "k8s-docs/scale-out", "k8s-docs/pod-eviction",
    "k8s-docs/resource-quota", "k8s-docs/network-policy",
    "horizontal_pod_autoscaler", "vertical_pod_autoscaler",
    "circuit_breaker", "retry_with_backoff", "load_shedding",
]


def build_synthetic_graph(uri: str, user: str, password: str, seed: int = 42):
    """向 Neo4j 注入合成节点和路径。

    计划：
    - 注入额外 Component/Symptom/Cause/Solution 节点
    - 创建 2/3/4 跳路径模板
    - 边的 valid_at/invalid_at 错开以支持时态剪枝
    - 不影响现有数据（用合成前缀标识）
    """
    rng = Random(seed)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    with driver.session() as s:
        # 1. 注入合成节点（用 synth 标记方便后续清理）
        for n in SYNTHETIC_COMPONENTS:
            s.run(
                "MERGE (c:Component {name: $n}) SET c.source = 'synth'",
                n=n,
            )
        for n in SYNTHETIC_SYMPTOMS:
            s.run(
                "MERGE (c:Symptom {name: $n}) SET c.source = 'synth', c.severity = 'warning'",
                n=n,
            )
        for n in SYNTHETIC_CAUSES:
            s.run(
                "MERGE (c:Cause {name: $n}) SET c.source = 'synth'",
                n=n,
            )
        for n in SYNTHETIC_SOLUTIONS:
            s.run(
                "MERGE (c:Solution {name: $n}) SET c.source = 'synth'",
                n=n,
            )

        # 2. 为每个 Component 关联 1-2 个 Symptom
        for comp in SYNTHETIC_COMPONENTS:
            n_symptoms = rng.randint(1, 2)
            symptoms = rng.sample(SYNTHETIC_SYMPTOMS, n_symptoms)
            t = base_time
            for sym in symptoms:
                t = t + timedelta(minutes=rng.randint(5, 30))
                s.run(
                    """
                    MATCH (c:Component {name: $comp})
                    MATCH (sym:Symptom {name: $sym_name})
                    MERGE (c)-[r:RELATES_TO {name: 'HAS_SYMPTOM'}]->(sym)
                    SET r.valid_at = datetime($va),
                        r.invalid_at = datetime($ia),
                        r.lag_seconds = $lag,
                        r.source = 'synth'
                    """,
                    comp=comp,
                    sym_name=sym,
                    va=t.isoformat(),
                    ia=(t + timedelta(hours=2)).isoformat(),
                    lag=0,
                )

        # 3. 为每个 Symptom 关联 1-2 个 Cause
        for sym in SYNTHETIC_SYMPTOMS:
            n_causes = rng.randint(1, 2)
            causes = rng.sample(SYNTHETIC_CAUSES, n_causes)
            t = base_time + timedelta(minutes=rng.randint(1, 10))
            for c in causes:
                t = t + timedelta(minutes=rng.randint(1, 5))
                s.run(
                    """
                    MATCH (sym:Symptom {name: $sym_name})
                    MATCH (c:Cause {name: $c_name})
                    MERGE (sym)-[r:RELATES_TO {name: 'CAUSED_BY'}]->(c)
                    SET r.valid_at = datetime($va),
                        r.invalid_at = datetime($ia),
                        r.lag_seconds = $lag,
                        r.source = 'synth'
                    """,
                    sym_name=sym,
                    c_name=c,
                    va=t.isoformat(),
                    ia=(t + timedelta(hours=1)).isoformat(),
                    lag=60,
                )

        # 4. 为每个 Cause 关联 1 个 Solution
        for c in SYNTHETIC_CAUSES:
            sol = rng.choice(SYNTHETIC_SOLUTIONS)
            t = base_time + timedelta(minutes=rng.randint(15, 30))
            s.run(
                """
                MATCH (c:Cause {name: $c_name})
                MATCH (sol:Solution {name: $sol_name})
                MERGE (c)-[r:RELATES_TO {name: 'RESOLVED_BY'}]->(sol)
                SET r.valid_at = datetime($va),
                    r.invalid_at = datetime($ia),
                    r.lag_seconds = $lag,
                    r.source = 'synth',
                    r.effectiveness = 0.8
                """,
                c_name=c,
                sol_name=sol,
                va=t.isoformat(),
                ia=(t + timedelta(hours=1)).isoformat(),
                lag=120,
            )

        # 5. 为部分 Cause 关联 1 个二级 Cause（形成 4 跳路径）
        for c1 in SYNTHETIC_CAUSES[:8]:
            # 找一个不同的 c2
            others = [c for c in SYNTHETIC_CAUSES if c != c1]
            c2 = rng.choice(others)
            t = base_time + timedelta(minutes=rng.randint(20, 40))
            s.run(
                """
                MATCH (c1:Cause {name: $c1_name})
                MATCH (c2:Cause {name: $c2_name})
                MERGE (c1)-[r:RELATES_TO {name: 'PROPAGATED_TO'}]->(c2)
                SET r.valid_at = datetime($va),
                    r.invalid_at = datetime($ia),
                    r.lag_seconds = $lag,
                    r.source = 'synth'
                """,
                c1_name=c1,
                c2_name=c2,
                va=t.isoformat(),
                ia=(t + timedelta(hours=1)).isoformat(),
                lag=180,
            )

        # 6. 验证
        for q, label in [
            ("MATCH (c:Component) WHERE c.source = 'synth' RETURN count(c) as cnt", "Synthetic Components"),
            ("MATCH (s:Symptom) WHERE s.source = 'synth' RETURN count(s) as cnt", "Synthetic Symptoms"),
            ("MATCH (c:Cause) WHERE c.source = 'synth' RETURN count(c) as cnt", "Synthetic Causes"),
            ("MATCH (s:Solution) WHERE s.source = 'synth' RETURN count(s) as cnt", "Synthetic Solutions"),
        ]:
            r = s.run(q).single()
            print(f"  {label}: {r['cnt']}")

        # 4 跳路径数
        r = s.run("""
            MATCH (comp:Component)-[r1:RELATES_TO]->(s:Symptom)-[r2:RELATES_TO]->(c1:Cause)-[r3:RELATES_TO]->(c2:Cause)-[r4:RELATES_TO]->(sol:Solution)
            WHERE r1.name = 'HAS_SYMPTOM' 
              AND r2.name IN ['CAUSED_BY', 'TRIGGERED_BY']
              AND r3.name IN ['CAUSED_BY', 'TRIGGERED_BY', 'PROPAGATED_TO']
              AND r4.name IN ['RESOLVED_BY', 'MITIGATED_BY']
            RETURN count(*) as cnt
        """).single()
        print(f"  4-hop 路径总数: {r['cnt']}")

    driver.close()
    print("Done.")


if __name__ == "__main__":
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD")
    build_synthetic_graph(uri, user, pwd)
