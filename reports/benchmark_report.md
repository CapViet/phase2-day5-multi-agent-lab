# Single-Agent vs Multi-Agent Benchmark

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline :: Research GraphRAG state-of-the-a | 43.51 | 0.0000 |  | 100% | 0% | sources=5 coverage=100% in_tok=323 out_tok=885 |
| multi :: Research GraphRAG state-of-the-a | 97.35 | 0.0000 |  | 100% | 0% | sources=5 coverage=100% in_tok=2100 out_tok=1303 |
| baseline :: Compare single-agent and multi-a | 121.25 | 0.0000 |  | 100% | 0% | sources=5 coverage=100% in_tok=286 out_tok=715 |
| multi :: Compare single-agent and multi-a | 76.23 | 0.0000 |  | 100% | 0% | sources=5 coverage=100% in_tok=1573 out_tok=838 |
| baseline :: Summarize production guardrails  | 39.87 | 0.0000 |  | 100% | 0% | sources=5 coverage=100% in_tok=296 out_tok=483 |
| multi :: Summarize production guardrails  | 90.43 | 0.0000 |  | 80% | 0% | sources=5 coverage=80% in_tok=1705 out_tok=1003 |

## Takeaway

Fastest run: **baseline :: Summarize production guardrails ** (39.87s). Best citation coverage: **baseline :: Summarize production guardrails ** (100%). Multi-agent typically trades higher latency/cost for better grounding and coverage; prefer it when the task decomposes cleanly and source attribution matters.
