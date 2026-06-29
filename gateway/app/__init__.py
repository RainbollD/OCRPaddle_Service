"""OCR comparison gateway.

A lightweight FastAPI service that accepts ``{model, path}`` requests, swaps the
requested model's vLLM backend container in (one at a time, since the server GPU
holds a single model), runs OCR via the backend's OpenAI-compatible API, and
writes the recognized text to a Markdown file on the server.
"""
