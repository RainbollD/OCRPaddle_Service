# DeepSeek-OCR — запуск

| Параметр | Значение |
|---|---|
| Имя в CLI | `deepseek-ocr` |
| HF repo | `deepseek-ai/DeepSeek-OCR` |
| Размер | ~6.7 ГБ BF16 |
| Порт | 8001 |
| API | http://localhost:8001 |

> В 16 ГБ VRAM влезает одна модель. Перед стартом останови другие бэкенды.
> Веса лежат в общем томе `ocr-hf-cache` — качаются один раз на все модели.

## 1. Запустить

```bash
cd backends/deepseek-ocr
docker compose up               # первый раз соберёт образ и скачает веса
# docker compose up -d          # в фоне
```

Понять, что работает:

```bash
docker compose logs -f                   # живой лог загрузки и запросов
docker compose ps                        # STATUS: (health: starting) -> (healthy)
curl -f http://localhost:8001/health     # 200 = модель готова
```

Готово, когда в логах `Application startup complete` / `Uvicorn running on …:8001`
и `docker compose ps` показывает `(healthy)`.

## 2. Прогнать файл

```bash
pip install -r ../../requirements.txt    # один раз
python ../../client/cli.py deepseek-ocr ../../data/scan.pdf
# -> output/deepseek-ocr/scan.md
```

## 3. Остановить

```bash
docker compose down              # остановить и удалить контейнер (том весов сохраняется)
docker compose stop              # просто остановить
```

## Голый curl (без CLI)

```bash
IMG=$(base64 -w0 page.png)
curl -s http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"deepseek-ai/DeepSeek-OCR",
  "messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$IMG"'"}},
    {"type":"text","text":"<image>\n<|grounding|>Convert the document to markdown. "}
  ]}],"max_tokens":16384,"temperature":0}'
```

## Особенности

- vLLM nightly **≥ 0.11.1** (на 0.11.0 DeepSeek-OCR не работает).
- Нужен NGram logits-процессор и отключённый prefix caching (см. `CMD` в [Dockerfile](Dockerfile)):
  `--logits-processors …:NGramPerReqLogitsProcessor`, `--no-enable-prefix-caching`.
- `--trust-remote-code` обязателен.
