# CV Platform — Video Analytics (LangGraph + Multimodal LLM)

Платформа анализа видео с оркестрацией через **LangGraph** и UI на **Streamlit** (режимы STANDARD/PRO).

## Запуск

1) Создайте окружение Python 3.11+

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Запуск приложения

```bash
streamlit run app.py
```

## Примечания по моделям

- **Local LLaVA**: `llava-hf/llava-v1.6-mistral-7b-hf` загружается лениво. Если модель не доступна/нет ресурсов, пайплайн автоматически попробует **GigaChat API** (если выбран).
- **CV-модели** (YOLO/ReID/track): подключены адаптерами; при отсутствии весов/зависимостей возвращают stub-контекст (пайплайн не падает).

## Выходные артефакты

Результаты сохраняются в `out/{timestamp}_{hash}/`:
- `answer.json` или `answer.txt`
- `events.parquet`
- `metadata.yaml`
- `cache/frame_hashes.pkl` (PRO + cache_frames)
- `processing_log.log`

