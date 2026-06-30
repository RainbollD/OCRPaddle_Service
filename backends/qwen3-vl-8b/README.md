# Qwen3-VL-8B-Instruct (FP8) — запуск

| Параметр | Значение |
|---|---|
| Имя в CLI | `qwen3-vl-8b` |
| HF repo | `Qwen/Qwen3-VL-8B-Instruct-FP8` |
| Размер | ~9 ГБ FP8 |
| Порт | 8003 |
| API | http://localhost:8003 |

> В 16 ГБ VRAM влезает одна модель. Перед стартом останови другие бэкенды.
> Веса лежат в общем томе `ocr-hf-cache` — качаются один раз на все модели.

## 1. Запустить

```bash
cd backends/qwen3-vl-8b
docker compose up               # первый раз соберёт образ и скачает веса
# docker compose up -d          # в фоне
```

Понять, что работает:

```bash
docker compose logs -f                   # живой лог загрузки и запросов
docker compose ps                        # STATUS: (health: starting) -> (healthy)
curl -f http://localhost:8003/health     # 200 = модель готова
```

Готово, когда в логах `Application startup complete` / `Uvicorn running on …:8003`
и `docker compose ps` показывает `(healthy)`.

## 2. Прогнать файл

```bash
pip install -r ../../requirements.txt    # один раз
python ../../client/cli.py qwen3-vl-8b ../../data/scan.pdf
# -> output/qwen3-vl-8b/scan.md
```

## 3. Остановить

```bash
docker compose down              # остановить и удалить контейнер (том весов сохраняется)
docker compose stop              # просто остановить
```

## Голый curl (без CLI)

```bash
IMG=$(base64 -w0 page.png)
curl -s http://localhost:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen/Qwen3-VL-8B-Instruct-FP8",
  "messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$IMG"'"}},
    {"type":"text","text":"Recognize all text in the image and output it as clean GitHub-flavored Markdown, preserving the original reading order, layout, tables (as Markdown tables) and formulas. The text is mostly Russian with some English. Output only the Markdown, with no commentary."}
  ]}],"max_tokens":16384,"temperature":0}'
```

## Особенности

- vLLM **≥ 0.11.0** для поддержки Qwen3-VL (на Blackwell sm_120 — `:nightly`).
- FP8-веса грузятся только под vLLM/SGLang (обычный transformers их не возьмёт).
- Видео-вход отключён ради экономии памяти: `--limit-mm-per-prompt '{"video": 0}'`
  (см. `CMD` в [Dockerfile](Dockerfile)).
