# HunyuanOCR backend (~1B, ~2 GB). Served via vLLM's OpenAI-compatible API.
#
# Requirements (verify against the actual RTX 5060 / Blackwell sm_120 host):
#   - vLLM nightly >= 0.12.0 (HunyuanOCR support + system-prompt fixes landed late 2025).
#   - Custom `hunyuan_vl` architecture -> needs --trust-remote-code.
#
# Pin to a known-good nightly digest once verified on the target host.
FROM vllm/vllm-openai:nightly

ENTRYPOINT []

CMD ["vllm", "serve", "tencent/HunyuanOCR", \
     "--host", "0.0.0.0", "--port", "8002", \
     "--gpu-memory-utilization", "0.9", \
     "--trust-remote-code", \
     "--no-enable-prefix-caching", \
     "--mm-processor-cache-gb", "0"]
