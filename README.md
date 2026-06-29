# OCR Comparison Stand

Стенд для сравнения OCR/VLM-моделей. Каждая модель — **отдельный контейнер** с vLLM
(OpenAI-совместимый API). Ты поднимаешь **один контейнер за раз** (в 16 ГБ VRAM влезает одна
модель) и обращаешься к нему. Готовый CLI-клиент рендерит PDF/картинку, шлёт в модель и пишет
`.md` с распознанным текстом. Никакого общего шлюза — прямое обращение к запущенному бэкенду.

Документы в основном на русском, встречается английский — у всех трёх моделей кириллица
поддерживается.

## Модели

| Модель (имя в CLI) | HF repo | Размер | Порт |
|---|---|---|---|
| `deepseek-ocr` | `deepseek-ai/DeepSeek-OCR` | ~6.7 ГБ BF16 | 8001 |
| `hunyuan-ocr` | `tencent/HunyuanOCR` | ~2 ГБ | 8002 |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct-FP8` | ~9 ГБ FP8 | 8003 |

Поддержка входа: изображения (`.png/.jpg/.jpeg/.webp/.bmp`) и `.pdf` (в т.ч. сканы — рендерятся
постранично, md склеивается через `---`).

## Требования

- GPU NVIDIA, **16 ГБ VRAM**; драйвер + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Docker ≥ 24 + Compose v2.
- Python 3.10+ для CLI (там, где запускаешь клиент): `pip install -r requirements.txt`.
- (опц.) `export HF_TOKEN=...` — быстрее качает веса с HF.

> Бэкенды используют образ `vllm/vllm-openai:nightly`. На стабильном железе (Ampere/Ada/Hopper)
> можно запиновать стабильный тег vLLM в `backends/<модель>/Dockerfile`.

## Запуск

### 1. Собрать образы (один раз)

```bash
docker compose build
```

### 2. Поднять ОДНУ модель

```bash
docker compose up deepseek-ocr        # или hunyuan-ocr / qwen3-vl-8b
```

Дождись в логах `Application startup complete` / `Uvicorn running on …:8001`. Проверка:

```bash
curl -f http://localhost:8001/health   # 200 = модель готова
```

> Первый старт качает веса и компилирует граф (`VLLM_COMPILE` + capture CUDA graphs) — это пара
> минут. Дальше старты быстрые из тома `hf-cache`.

### 3. Прогнать файл через CLI

```bash
pip install -r requirements.txt        # один раз

python client/cli.py deepseek-ocr data/scan.pdf
# -> output/deepseek-ocr/scan.md
```

### 4. Сравнить с другой моделью

Останови текущий контейнер и подними следующий (в 16 ГБ две сразу не влезут):

```bash
docker compose stop deepseek-ocr
docker compose up hunyuan-ocr
python client/cli.py hunyuan-ocr data/scan.pdf
# -> output/hunyuan-ocr/scan.md
```

Так результаты по каждой модели ложатся в свою папку `output/<модель>/<имя>.md` — удобно сравнивать.

## CLI

```text
python client/cli.py <model> <file> [options]

  <model>          deepseek-ocr | hunyuan-ocr | qwen3-vl-8b
  <file>           путь к картинке или PDF (локальный путь там, где запускаешь CLI)

  --url URL        база бэкенда (по умолчанию http://localhost:<порт модели>)
  --host HOST      хост для дефолтного URL (по умолчанию localhost)
  --out DIR        корень для вывода (по умолчанию ./output)
  --dpi N          DPI рендера PDF (по умолчанию 200; для плотных сканов лучше 300)
  --max-pages N    лимит страниц PDF (по умолчанию 100)
  --timeout SEC    таймаут на страницу (по умолчанию 600)
```

### Доступ с другой машины (SSH-туннель)

CLI можно запускать прямо на сервере, либо локально через туннель к нужному порту:

```bash
ssh -L 8001:localhost:8001 user@server          # пробросить порт модели
python client/cli.py deepseek-ocr ./scan.pdf     # CLI шлёт на localhost:8001
```

### Без CLI (один image, голый curl)

```bash
IMG=$(base64 -w0 page.png)
curl -s http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"deepseek-ai/DeepSeek-OCR",
  "messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$IMG"'"}},
    {"type":"text","text":"<image>\n<|grounding|>Convert the document to markdown. "}
  ]}],"max_tokens":16384,"temperature":0}'
```

## Добавить модель

1. `backends/<имя>/Dockerfile` (база `vllm/vllm-openai:nightly`, свой `vllm serve … --port <порт>`).
2. Сервис в `docker-compose.yml` (GPU, том `hf-cache`, проброс порта).
3. `ModelSpec` в `client/registry.py` (имя, порт, `model_id`, промпт).
