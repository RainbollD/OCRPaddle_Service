# Qwen3-VL-8B-Instruct backend, FP8 (~9 GB). Served via vLLM's OpenAI API.
# FP8 weights load only under vLLM/SGLang (not plain transformers).
#
# Requirements (verify against the actual RTX 5060 / Blackwell sm_120 host):
#   - vLLM >= 0.11.0 for Qwen3-VL support (use :nightly for Blackwell kernels).
#   - Disable video inputs (--limit-mm-per-prompt) to save memory; OCR is images only.
#
# Pin to a known-good nightly digest once verified on the target host.
FROM vllm/vllm-openai:nightly

ENTRYPOINT []

CMD ["vllm", "serve", "Qwen/Qwen3-VL-8B-Instruct-FP8", \
     "--host", "0.0.0.0", "--port", "8003", \
     "--gpu-memory-utilization", "0.9", \
     "--limit-mm-per-prompt", "{\"video\": 0}"]
