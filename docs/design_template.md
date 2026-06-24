# Design: Multi-Agent Research System

## Problem

Trả lời một câu hỏi nghiên cứu dài (vd "Research GraphRAG state-of-the-art and write a 500-word
summary"): cần tìm nguồn, phân tích, và viết câu trả lời có trích dẫn cho đối tượng kỹ thuật.

## Why multi-agent?

Một single-agent phải vừa search, vừa lọc nguồn, vừa phân tích, vừa viết trong một context — dễ
bỏ sót bước, khó kiểm soát chất lượng từng phần, và khó debug khi sai. Tách vai trò cho phép mỗi
agent có một system prompt hẹp, dễ trace, và dễ thay từng phần (vd đổi search provider) mà không
ảnh hưởng phần khác. Đánh đổi: latency và token cao hơn (xem benchmark).

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Routing rule-based + guardrail | state hiện tại | `next_action` | Loop vô hạn → chặn bằng `max_iterations` |
| Researcher | Search + viết research notes có [n] | query | `sources`, `research_notes` | Search lỗi/0 nguồn → `AgentExecutionError`, ghi error |
| Analyst | Trích claim, so sánh, flag bằng chứng yếu | `research_notes` | `analysis_notes` | Thiếu notes → ghi error, trả state |
| Writer | Tổng hợp câu trả lời + Sources list | research + analysis | `final_answer` | Thiếu input → ghi error |
| Critic (optional) | Citation coverage + review | `final_answer` | findings + coverage trong trace | Không có answer → ghi error |

## Shared state (`core/state.py`)

- `request`: câu hỏi + cấu hình (audience, max_sources).
- `route_history`, `iteration`, `next_action`: phục vụ routing & guardrail, debug được luồng.
- `sources`, `research_notes`, `analysis_notes`, `final_answer`: outputs để handoff giữa agent.
- `total_input_tokens` / `total_output_tokens` / `total_cost_usd`: cost accounting gộp.
- `agent_results`, `trace`, `errors`: audit từng bước (ai làm gì, tốn bao nhiêu, sai ở đâu).

## Routing policy

```text
supervisor --(no research_notes)--> researcher --> supervisor
supervisor --(no analysis_notes)--> analyst    --> supervisor
supervisor --(no final_answer)----> writer     --> supervisor [--> critic --> supervisor]
supervisor --(answer done | iteration>=max | errors>=max)--> DONE
```

Triển khai bằng LangGraph `StateGraph` (khi cài được); nếu LangGraph trong môi trường lỗi/thiếu,
một driver tương đương trong `MultiAgentWorkflow._run_without_langgraph` chạy cùng routing đó.

## Guardrails

- **Max iterations**: `Settings.max_iterations` (default 6) — supervisor trả `done` khi chạm trần.
- **Timeout**: `Settings.timeout_seconds` truyền vào LLM client.
- **Retry**: `tenacity` 3 lần (exponential backoff) cho mọi LLM call trong `LLMClient`.
- **Fallback**: không có key → offline mock; Tavily lỗi → mock search; LangGraph lỗi → built-in driver.
- **Validation**: tất cả I/O qua Pydantic schema; benchmark cô lập từng run, run lỗi = failure_rate 1.0.

## Benchmark plan

- Queries: xem `configs/lab_default.yaml`.
- Metrics: latency (wall-clock), cost (token × pricing; Ollama = 0), citation coverage (số [n] hợp
  lệ / số nguồn), failure rate.
- Expected: multi-agent latency/cost cao hơn baseline, đổi lại grounding/coverage tốt hơn khi task
  tách bước rõ ràng. Kết quả thực tế: `reports/benchmark_report.md`.

## Failure modes quan sát được & cách fix

1. **Loop không dừng** nếu một worker liên tục fail (notes vẫn `None`). Fix: supervisor đếm
   `errors` và dừng khi `errors >= max_iterations`, cộng thêm trần `iteration`.
2. **Citation coverage giả 100%** vì regex `[n]` quét cả khối "Sources:" auto-append. Fix khả dĩ:
   tính coverage chỉ trên phần thân trước "Sources:". Hiện chấp nhận vì cả 2 run xử lý như nhau.
3. **Model không tồn tại trên Ollama** (vd `llama3.1` chưa pull) → 404. Fix: đổi `OLLAMA_MODEL`
   sang model đã pull (`ollama list`), hoặc `LLM_PROVIDER=mock`.

## Exit ticket

1. **Nên dùng multi-agent khi nào?** Khi task tách được thành các bước rõ ràng với tiêu chí chất
   lượng khác nhau (tìm nguồn vs phân tích vs viết), khi cần trace/audit từng bước, và khi grounding
   + trích dẫn quan trọng hơn tốc độ.
2. **Không nên dùng khi nào?** Khi task đơn giản/một bước, latency và cost là ràng buộc chính, hoặc
   khi việc chia vai trò chỉ thêm overhead mà không cải thiện chất lượng — single-agent baseline
   cho kết quả tương đương nhanh và rẻ hơn (xem cột latency/cost trong benchmark).
