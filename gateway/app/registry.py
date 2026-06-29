from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """Describes one OCR backend the gateway can drive.

    Attributes:
        name: Public model name used in API requests (and output subfolder).
        container: Docker container name of the backend (set in compose).
        service: Compose service / network hostname the gateway calls.
        port: Port the backend's OpenAI-compatible server listens on.
        model_id: The ``model`` value sent in the chat-completions request
            (must match how vLLM serves the model).
        prompt: OCR instruction sent alongside the image.
        max_tokens: Generation cap for one image.
        temperature: Sampling temperature (0 = deterministic, best for OCR).
        extra_body: Extra fields merged into the chat-completions payload
            (e.g. ``chat_template_kwargs`` or vLLM-specific options).
    """

    name: str
    container: str
    service: str
    port: int
    model_id: str
    prompt: str
    max_tokens: int = 16384
    temperature: float = 0.0
    extra_body: dict = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        """Return the backend's OpenAI-compatible base URL on the docker network."""
        return f"http://{self.service}:{self.port}"


# Reference prompts come from each model's official docs (see README / plan).
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
        container="ocr-deepseek",
        service="deepseek-ocr",
        port=8001,
        model_id="deepseek-ai/DeepSeek-OCR",
        prompt=_DEEPSEEK_PROMPT,
    ),
    "hunyuan-ocr": ModelSpec(
        name="hunyuan-ocr",
        container="ocr-hunyuan",
        service="hunyuan-ocr",
        port=8002,
        model_id="tencent/HunyuanOCR",
        prompt=_HUNYUAN_PROMPT,
    ),
    "qwen3-vl-8b": ModelSpec(
        name="qwen3-vl-8b",
        container="ocr-qwen3-vl-8b",
        service="qwen3-vl-8b",
        port=8003,
        model_id="Qwen/Qwen3-VL-8B-Instruct-FP8",
        prompt=_QWEN_PROMPT,
    ),
}


def model_names() -> list[str]:
    """Return the configured model names in a stable order."""
    return list(MODELS.keys())


def get_model(name: str) -> ModelSpec:
    """Look up a :class:`ModelSpec` by public name.

    Args:
        name: Public model name from an API request.

    Returns:
        The matching :class:`ModelSpec`.

    Raises:
        KeyError: If ``name`` is not a configured model.
    """
    return MODELS[name]
