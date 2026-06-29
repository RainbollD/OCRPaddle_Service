# DeepSeek-OCR backend (~6.7 GB BF16). Served via vLLM's OpenAI-compatible API.
#
# Requirements (verify against the actual RTX 5060 / Blackwell sm_120 host):
#   - vLLM nightly >= 0.11.1 (plain 0.11.0 does NOT work for DeepSeek-OCR).
#   - DeepSeek-OCR needs an NGram logits processor and prefix caching disabled.
#
# The :nightly tag is used because Blackwell (sm_120) needs recent CUDA 12.8+
# kernels. Pin to a known-good digest once verified on the target host.
FROM vllm/vllm-openai:nightly

# vLLM's official image sets an entrypoint; override so CMD is the full command.
ENTRYPOINT []

CMD ["vllm", "serve", "deepseek-ai/DeepSeek-OCR", \
     "--host", "0.0.0.0", "--port", "8001", \
     "--gpu-memory-utilization", "0.9", \
     "--trust-remote-code", \
     "--logits-processors", "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor", \
     "--no-enable-prefix-caching", \
     "--mm-processor-cache-gb", "0"]
