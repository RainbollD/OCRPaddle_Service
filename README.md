# OCR Comparison Stand

Стенд для сравнения OCR/VLM-моделей. На сервере поднимается контейнер-**гейтвей**, к которому
идут API-запросы вида «модель + путь к файлу». Гейтвей подгружает нужную модель (один
vLLM-бэкенд за раз — в 16 ГБ VRAM влезает одна модель), прогоняет картинку/PDF и **пишет
`.md`-файл** с распознанным текстом на сервер. Режим `all` гоняет файл через все модели по очереди.

Документы в основном на русском, встречается английский — у всех трёх моделей кириллица в
списке поддерживаемых языков.

## Архитектура

```
client ──SSH-туннель──▶ gateway (FastAPI, :8080, CPU)
                          │  docker.sock: start/stop бэкендов
                          │  HTTP /v1/chat/completions
                          ▼
              один из vLLM-бэкендов (GPU, OpenAI API)
   ┌───────────────┬──────────────┬──────────────────────┐
   deepseek-ocr    hunyuan-ocr     qwen3-vl-8b
   :8001           :8002           :8003
```

| Модель | HF repo | Размер | Порт |
|---|---|---|---|
| `deepseek-ocr` | `deepseek-ai/DeepSeek-OCR` | ~6.7 ГБ BF16 | 8001 |
| `hunyuan-ocr` | `tencent/HunyuanOCR` | ~2 ГБ | 8002 |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct-FP8` | ~9 ГБ FP8 | 8003 |

- **Вход:** файл лежит в `./data` (монтируется в гейтвей как `/data`), запрос передаёт путь
  относительно этой папки. Поддержка изображений (`.png/.jpg/.jpeg/.webp/.bmp`) и `.pdf`
  (рендерится постранично, md склеивается через `---`).
- **Выход:** `./output/<модель>/<имя>.md`.

## Требования

| Требование | Примечание |
|---|---|
| GPU NVIDIA, **16 ГБ VRAM** (RTX 5060 / Blackwell, sm_120) | держит одну модель за раз |
| Драйвер NVIDIA + **CUDA 12.8+** | нужно для Blackwell-ядер vLLM |
| Docker ≥ 24 + Compose v2 | `docker compose version` |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | проброс GPU в контейнер |
| (опц.) `HF_TOKEN` | если модель требует токен; экспортируется как env |

> ⚠️ **Главный риск — тулчейн под Blackwell.** Бэкенды используют образ `vllm/vllm-openai:nightly`.
> Сначала пройдите **smoke-тест железа** (ниже). Если vLLM не стартует под sm_120 — поменяйте тег
> образа / версию в `backends/<модель>/Dockerfile`, прежде чем идти дальше.

## Запуск

### 0. Smoke-тест железа (делать первым, на сервере с GPU)

```bash
nvidia-smi                                  # CUDA >= 12.8, видна RTX 5060
docker compose --profile backend build deepseek-ocr
docker compose --profile backend up deepseek-ocr     # дождаться загрузки модели
curl -f http://localhost:8001/health                 # ожидаем 200
# ручной запрос к бэкенду (картинку base64 подставить) — см. vLLM OpenAI API
docker compose stop deepseek-ocr
```

Повторить для `hunyuan-ocr` (:8002) и `qwen3-vl-8b` (:8003) — **по очереди**, в 16 ГБ
одновременно влезает одна модель.

### 1. Сборка

```bash
docker compose build gateway
docker compose --profile backend build      # три бэкенда
```

### 2. Создать (но не запускать) контейнеры бэкендов

Гейтвей стартует/гасит бэкенды по имени, поэтому контейнеры должны существовать:

```bash
docker compose --profile backend up --no-start
```

### 3. Поднять гейтвей

```bash
docker compose up -d gateway
```

### 4. Доступ с локальной машины через SSH-туннель

```bash
ssh -L 8080:localhost:8080 user@server
```

Дальше все `curl` — на `http://localhost:8080`.

## API

### `GET /health`
```bash
curl http://localhost:8080/health           # {"status":"ok"}
```

### `GET /models`
```bash
curl http://localhost:8080/models            # список моделей + какая сейчас активна
```

### `POST /ocr`
```bash
# одна модель
curl -X POST http://localhost:8080/ocr \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ocr","path":"sample_ru.png"}'

# все модели по очереди (load/swap между ними)
curl -X POST http://localhost:8080/ocr \
  -H 'Content-Type: application/json' \
  -d '{"model":"all","path":"sample_ru.png"}'
```

Ответ:
```json
{
  "input": "sample_ru.png",
  "results": [
    { "model": "deepseek-ocr", "status": "ok",
      "output_path": "output/deepseek-ocr/sample_ru.md",
      "chars": 1234, "pages": 1, "elapsed_s": 8.1 }
  ]
}
```

Поля запроса:

| Поле | Тип | По умолч. | Описание |
|---|---|---|---|
| `model` | str | — | имя модели из `/models` или `"all"` |
| `path` | str | — | путь к файлу относительно `./data` |
| `keep_loaded` | bool | `true` | не гасить бэкенд после запроса (быстрее повторные вызовы) |

Результат каждой модели лежит в `./output/<модель>/<имя>.md`.

## Переменные окружения (гейтвей)

| Переменная | Default | Описание |
|---|---|---|
| `DATA_DIR` | `/data` | папка входных файлов внутри контейнера |
| `OUTPUT_DIR` | `/output` | папка для `.md` |
| `DOCKER_NETWORK` | `ocr-net` | docker-сеть гейтвея и бэкендов |
| `BACKEND_READY_TIMEOUT_S` | `900` | сколько ждать готовности бэкенда (первый старт = скачивание весов) |
| `OCR_REQUEST_TIMEOUT_S` | `600` | таймаут генерации на запрос |
| `PDF_DPI` | `200` | разрешение рендера PDF |
| `MAX_PDF_PAGES` | `100` | лимит страниц PDF |

## Добавить модель

1. Создать `backends/<имя>/Dockerfile` (база `vllm/vllm-openai:nightly`, свой `vllm serve ...`).
2. Добавить сервис в `docker-compose.yml` (profile `backend`, GPU, том `hf-cache`, свой порт).
3. Добавить `ModelSpec` в `gateway/app/registry.py` (имя, контейнер, сервис, порт, `model_id`, промпт).
