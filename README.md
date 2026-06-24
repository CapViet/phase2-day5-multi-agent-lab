# Lab 20: Multi-Agent Research System Starter

Starter repo cho bài lab **Multi-Agent Systems**: xây dựng hệ thống nghiên cứu gồm **Supervisor + Researcher + Analyst + Writer** và benchmark với single-agent baseline.

> Mục tiêu của repo này là cung cấp **production-grade skeleton** để học viên phát triển code cá nhân. Các phần logic quan trọng được để ở dạng `TODO` để học viên tự triển khai.

## Learning outcomes

Sau 2 giờ lab, học viên cần có thể:

1. Thiết kế role rõ ràng cho nhiều agent.
2. Xây dựng shared state đủ thông tin cho handoff.
3. Thêm guardrail tối thiểu: max iterations, timeout, retry/fallback, validation.
4. Trace được luồng chạy và giải thích agent nào làm gì.
5. Benchmark single-agent vs multi-agent theo quality, latency, cost.

## Architecture mục tiêu

```text
User Query
   |
   v
Supervisor / Router
   |------> Researcher Agent  -> research_notes
   |------> Analyst Agent     -> analysis_notes
   |------> Writer Agent      -> final_answer
   |
   v
Trace + Benchmark Report
```

## Cấu trúc repo

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/              # Agent interfaces + skeletons
│   ├── core/                # Config, state, schemas, errors
│   ├── graph/               # LangGraph workflow skeleton
│   ├── services/            # LLM, search, storage clients
│   ├── evaluation/          # Benchmark/evaluation skeleton
│   ├── observability/       # Logging/tracing hooks
│   └── cli.py               # CLI entrypoint
├── configs/                 # YAML configs for lab variants
├── docs/                    # Lab guide, rubric, design notes
├── tests/                   # Unit tests for skeleton behavior
├── notebooks/               # Optional notebook entrypoint
├── scripts/                 # Helper scripts
├── .env.example             # Environment variables template
├── pyproject.toml           # Python project config
├── Dockerfile               # Containerized dev/runtime
└── Makefile                 # Common commands
```

## Quickstart

### 1. Tạo môi trường

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev,llm]"
cp .env.example .env
```

### 2. Chọn LLM backend (switchable)

Hệ thống chạy được **ngay cả khi không có API key** nhờ offline mock deterministic. Chọn backend qua `LLM_PROVIDER`:

| `LLM_PROVIDER` | Dùng khi | Model switch bằng |
|---|---|---|
| `auto` (default) | Tự chọn: OpenAI nếu có key, else Ollama local nếu chạy, else mock | — |
| `openai` | Có `OPENAI_API_KEY` | `OPENAI_MODEL` |
| `ollama` | Có Ollama chạy local (`ollama serve`) | `OLLAMA_MODEL` (vd `llama3.1`, `qwen2.5:3b`) |
| `mock` | Không cần key/mạng (CI, test) | — |

Ví dụ chạy bằng model local khác chỉ cần đổi 1 biến:

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=qwen2.5:3b   # đổi model ở đây
```

`.env` mẫu:

```bash
LLM_PROVIDER=auto
OPENAI_API_KEY=...          # optional
OLLAMA_MODEL=llama3.1       # optional, cho Ollama
TAVILY_API_KEY=...          # optional, nếu không có sẽ dùng mock search
```

### 3. Chạy smoke test

```bash
make test
python -m multi_agent_research_lab.cli --help
```

### 4. Chạy single-agent baseline

```bash
python -m multi_agent_research_lab.cli baseline \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Một agent làm toàn bộ trong một LLM call (control để benchmark).

### 5. Chạy multi-agent workflow

```bash
python -m multi_agent_research_lab.cli multi-agent \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary" \
  --critic --trace-out reports/trace.json
```

Supervisor điều phối Researcher → Analyst → Writer (→ Critic nếu `--critic`), in route, token/cost, và export JSON trace.

### 6. Benchmark single vs multi-agent

```bash
python -m multi_agent_research_lab.cli benchmark \
  --query "Summarize production guardrails for LLM agents"
```

Đo latency, cost, citation coverage, failure rate cho cả hai cách và ghi `reports/benchmark_report.md`. Không truyền `--query` sẽ chạy bộ query mặc định.

## Milestones trong 2 giờ lab

| Thời lượng | Milestone | File gợi ý |
|---:|---|---|
| 0-15' | Setup, chạy baseline skeleton | `cli.py`, `services/llm_client.py` |
| 15-45' | Build Supervisor / router | `agents/supervisor.py`, `graph/workflow.py` |
| 45-75' | Thêm Researcher, Analyst, Writer | `agents/*.py`, `core/state.py` |
| 75-95' | Trace + benchmark single vs multi | `observability/tracing.py`, `evaluation/benchmark.py` |
| 95-115' | Peer review theo rubric | `docs/peer_review_rubric.md` |
| 115-120' | Exit ticket | `docs/lab_guide.md` |

## Quy ước production trong repo

- Tách rõ `agents`, `services`, `core`, `graph`, `evaluation`, `observability`.
- Không hard-code API key trong code.
- Tất cả input/output chính dùng Pydantic schema.
- Có type hints, linting, formatting, unit test tối thiểu.
- Có logging/tracing hook ngay từ đầu.
- Không để agent chạy vô hạn: dùng `max_iterations`, `timeout_seconds`.
- Có benchmark report thay vì chỉ demo output đẹp.

## TODO chính cho học viên

Tìm trong code các marker:

```bash
grep -R "TODO(student)" -n src tests docs
```

Các phần học viên cần tự làm:

1. Implement LLM client.
2. Implement web/search client hoặc mock search source.
3. Implement routing decision trong Supervisor.
4. Implement từng worker agent.
5. Build LangGraph workflow.
6. Thêm tracing provider thật: LangSmith, Langfuse hoặc OpenTelemetry.
7. Viết benchmark report.

## Deliverables

Học viên nộp:

1. GitHub repo cá nhân.
2. Screenshot trace hoặc link trace.
3. `reports/benchmark_report.md` so sánh single vs multi-agent.
4. Một đoạn giải thích failure mode và cách fix.

## References

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Agents SDK orchestration/handoffs — https://developers.openai.com/api/docs/guides/agents/orchestration
- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- Langfuse tracing — https://langfuse.com/docs
