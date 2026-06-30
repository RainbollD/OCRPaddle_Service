# HunyuanOCR — запуск

| Параметр | Значение |
|---|---|
| Имя в CLI | `hunyuan-ocr` |
| HF repo | `tencent/HunyuanOCR` |
| Размер | ~1B, ~2 ГБ |
| Порт | 8002 |
| API | http://localhost:8002 |

> В 16 ГБ VRAM влезает одна модель. Перед стартом останови другие бэкенды.
> Веса лежат в общем томе `ocr-hf-cache` — качаются один раз на все модели.

## 1. Запустить

```bash
cd backends/hunyuan-ocr
docker compose up               # первый раз соберёт образ и скачает веса
# docker compose up -d          # в фоне
```

Понять, что работает:

```bash
docker compose logs -f                   # живой лог загрузки и запросов
docker compose ps                        # STATUS: (health: starting) -> (healthy)
curl -f http://localhost:8002/health     # 200 = модель готова
```

Готово, когда в логах `Application startup complete` / `Uvicorn running on …:8002`
и `docker compose ps` показывает `(healthy)`.

## 2. Прогнать файл

```bash
pip install -r ../../requirements.txt    # один раз
python ../../client/cli.py hunyuan-ocr ../../data/scan.pdf
# -> output/hunyuan-ocr/scan.md
```

## 3. Остановить

```bash
docker compose down              # остановить и удалить контейнер (том весов сохраняется)
docker compose stop              # просто остановить
```

## Голый curl (без CLI)

Рекомендованный промпт для разбора документа (мультиязычный, кириллица ок): текст → Markdown,
таблицы → HTML, формулы → LaTeX.

```bash
IMG=$(base64 -w0 page.png)
curl -s http://localhost:8002/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"tencent/HunyuanOCR",
  "messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$IMG"'"}},
    {"type":"text","text":"提取文档图片中正文的所有信息用markdown格式表示，表格用html格式表达，文档中公式用latex格式表示"}
  ]}],"max_tokens":16384,"temperature":0}'
```

## Особенности

- vLLM nightly **≥ 0.12.0** (поддержка HunyuanOCR + фиксы system-prompt появились в конце 2025).
- Кастомная архитектура `hunyuan_vl` → нужен `--trust-remote-code`.
- Prefix caching отключён (см. `CMD` в [Dockerfile](Dockerfile)).
