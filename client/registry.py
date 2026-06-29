from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Describes one OCR backend the CLI can talk to.

    Attributes:
        name: Short model name used on the command line (and output subfolder).
        port: Port the backend's vLLM OpenAI server publishes on the host.
        model_id: The ``model`` value sent in the chat-completions request
            (must match how vLLM serves the model).
        prompt: OCR instruction sent alongside the image.
        max_tokens: Generation cap for one image.
        temperature: Sampling temperature (0 = deterministic, best for OCR).
    """

    name: str
    port: int
    model_id: str
    prompt: str
    max_tokens: int = 16384
    temperature: float = 0.0

    def default_url(self, host: str = "localhost") -> str:
        """Return the backend's base URL from its published port."""
        return f"http://{host}:{self.port}"


# Reference prompts come from each model's official docs (see README).
_DEEPSEEK_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "

# HunyuanOCR's recommended document-parsing prompt (works multilingually,
# including Russian/Cyrillic): body text -> Markdown, tables -> HTML, formulas -> LaTeX.
_HUNYUAN_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，"
    "表格用html格式表达，文档中公式用latex格式表示"
)

_QWEN_PROMPT = (
    "Recognize all text in the image and output it as clean GitHub-flavored "
    "Markdown, preserving the original reading order, layout, tables (as Markdown "
    "tables) and formulas. The text is mostly Russian with some English. "
    "Output only the Markdown, with no commentary."
)


MODELS: dict[str, ModelSpec] = {
    "deepseek-ocr": ModelSpec(
        name="deepseek-ocr",
        port=8001,
        model_id="deepseek-ai/DeepSeek-OCR",
        prompt=_DEEPSEEK_PROMPT,
    ),
    "hunyuan-ocr": ModelSpec(
        name="hunyuan-ocr",
        port=8002,
        model_id="tencent/HunyuanOCR",
        prompt=_HUNYUAN_PROMPT,
    ),
    "qwen3-vl-8b": ModelSpec(
        name="qwen3-vl-8b",
        port=8003,
        model_id="Qwen/Qwen3-VL-8B-Instruct-FP8",
        prompt=_QWEN_PROMPT,
    ),
}
