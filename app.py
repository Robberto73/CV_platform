from __future__ import annotations

import os
import time
import uuid
from typing import Any

import pandas as pd
import streamlit as st
import hashlib

from pipeline.config import (
    describe_loaded_config_for_ui,
    load_app_settings,
    load_gigachat_ca_bundle_file,
    load_gigachat_default_key,
    save_app_settings,
)
from pipeline.run_pipeline import run_pipeline
from pipeline.state import PipelineState


st.set_page_config(page_title="CV Platform — Video Analytics", layout="wide")


def _save_uploads(files: list[Any]) -> list[str]:
    session_id = str(uuid.uuid4())
    base = os.path.join(".tmp_uploads", session_id)
    os.makedirs(base, exist_ok=True)
    out_paths: list[str] = []
    for f in files:
        path = os.path.join(base, f.name)
        with open(path, "wb") as w:
            w.write(f.getbuffer())
        out_paths.append(path)
    return out_paths


st.title("Платформа анализа видео (LangGraph + Multimodal LLM)")

with st.sidebar:
    st.header("Настройки")

    cfg_info = describe_loaded_config_for_ui()
    with st.expander("Config (debug)", expanded=False):
        st.write(
            {
                "llava_model_id": cfg_info["llava_model_id"],
                "yolo_model_path": cfg_info["yolo_model_path"],
                "osnet_reid_model": cfg_info["osnet_reid_model"],
                "gigachat_available": cfg_info["gigachat_key_present"],
            }
        )

    ui_mode = st.radio("Режим", ["STANDARD", "PRO"], index=0, horizontal=True)
    force_no_visual_analysis = st.checkbox(
        "CV-only (без LLaVA/GigaChat)",
        value=False,
        help="Полезно для слабого ПК: пропускает мультимодальную LLM, но оставляет CV обработку и сохранение результатов.",
    )

    app_settings = load_app_settings()
    analyze_people = st.checkbox(
        "Анализ людей (YOLOv8)",
        value=bool(app_settings.get("analyze_people_default", True)),
        help="Если включено, детекция людей будет активна по умолчанию (yolov8n.pt).",
    )

    # Автоматически включаем анализ людей, если выбрано сохранение уникальных людей
    save_unique_people_enabled = False
    if ui_mode == "PRO":
        save_unique_people_enabled = st.session_state.get("save_unique_people", False) if "save_unique_people" in st.session_state else False

    if save_unique_people_enabled and not analyze_people:
        analyze_people = True
        st.info("⚠ Анализ людей автоматически включен для сохранения уникальных людей")
    analyze_pose = st.checkbox(
        "Анализ позы и движений (MediaPipe)",
        value=False,
        help="Если включено, будет выполняться анализ движений рук, ног и позы людей для более детального понимания действий.",
    )
    if st.button("Сохранить настройки по умолчанию"):
        app_settings["analyze_people_default"] = bool(analyze_people)
        save_app_settings(app_settings)
        st.success("Сохранено в config/app_settings.yaml")

    pro_settings = {
        "frame_sampling_rate": 1.0,
        "ssim_threshold": 0.85,  # Изменено с 0.9 на 0.85 для меньшего отбрасывания кадров
        "skip_static_frames": True,
        "cache_frames": True,
        "custom_preprocessing": "None",
    }
    if ui_mode == "PRO":
        pro_settings["frame_sampling_rate"] = st.slider(
            "Частота кадров (кадр/сек)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Количество кадров в секунду для анализа. Значение 1.0 = каждый кадр, 0.5 = каждый второй кадр. Высокие значения дают больше данных, но медленнее обрабатываются.",
        )
        pro_settings["ssim_threshold"] = st.slider(
            "Порог статичности SSIM",
            min_value=0.8,
            max_value=0.99,
            value=0.85,  # Изменено с 0.9 на 0.85
            step=0.01,
            help="Порог схожести кадров для определения статичных сцен. Более высокие значения (0.95+) пропускают только почти идентичные кадры. Более низкие (0.85-) пропускают больше кадров, но могут потерять важные изменения.",
        )
        pro_settings["skip_static_frames"] = st.checkbox(
            "Пропускать статичные кадры",
            value=True,
            help="Автоматически пропускать похожие кадры в статичных сценах для ускорения обработки. Сохраняет качество анализа динамичных моментов.",
        )
        pro_settings["cache_frames"] = st.checkbox(
            "Кэшировать кадры",
            value=True,
            help="Сохранять обработанные кадры на диск для повторного использования. Ускоряет повторные анализы того же видео, но требует дополнительного места на диске.",
        )
        pro_settings["custom_preprocessing"] = st.selectbox(
            "Дополнительная обработка",
            ["None", "Blur Detection (stub)", "Motion Emphasis (stub)"],
            index=0,
            help="Дополнительные алгоритмы предобработки кадров. 'Blur Detection' и 'Motion Emphasis' пока не реализованы - заглушки для будущих функций.",
        )

        # YOLO PRO settings
        with st.expander("YOLO настройки (PRO)", expanded=False):
            pro_settings["yolo_input_size"] = st.slider(
                "Размер входного изображения YOLO",
                min_value=320,
                max_value=1280,
                value=640,
                step=64,
                help="Размер входного изображения для YOLO модели в пикселях. Большие значения (640-1280) дают более точное обнаружение, но требуют больше памяти GPU и работают медленнее. Маленькие значения (320-512) быстрее, но менее точны.",
            )
            pro_settings["yolo_conf_threshold"] = st.slider(
                "Порог уверенности YOLO",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Минимальный уровень уверенности для детекции объектов YOLO. Более высокие значения (0.6-0.8) уменьшают ложные срабатывания, но могут пропустить объекты. Более низкие (0.3-0.5) находят больше объектов, но увеличивают шум.",
            )
        pro_settings["reid_event_mode"] = st.selectbox(
            "ReID режим детализации",
            ["segments (рекомендуется)", "frames (детально)"],
            index=0,
            help="'Segments' - компактные события по остановкам/перемещениям (быстрее, меньше данных). 'Frames' - события по каждому кадру (детально, но больше данных и медленнее).",
        )
        # нормализуем в внутреннее значение
        pro_settings["reid_event_mode"] = "segments" if "segments" in pro_settings["reid_event_mode"] else "frames"
        if pro_settings["reid_event_mode"] == "frames":
            pro_settings["reid_frames_min_dt_sec"] = st.slider(
                "ReID мин. интервал событий (сек)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                help="Минимальный временной интервал между событиями позиции одной персоны. Большие значения (1-2 сек) ускоряют обработку, уменьшают размер выходных данных.",
            )
            pro_settings["reid_frames_max_points_per_person"] = st.number_input(
                "ReID макс. точек на персону",
                min_value=50,
                max_value=20000,
                value=2000,
                step=50,
                help="Максимальное количество позиционных событий для одной персоны. Защищает от переполнения данными для часто появляющихся людей.",
            )
            pro_settings["reid_frames_max_total_events"] = st.number_input(
                "ReID макс. общих событий",
                min_value=500,
                max_value=200000,
                value=20000,
                step=500,
                help="Общий лимит всех ReID событий. Предотвращает создание огромных файлов events.parquet при анализе длинных видео с множеством людей.",
            )

        # Unique people photos saving
        with st.expander("Сохранение уникальных людей (PRO)", expanded=False):
            pro_settings["save_unique_people"] = st.checkbox(
                "Сохранять уникальных людей",
                value=False,
                help="Автоматически сохранять лучшие фото каждого уникального человека из видео. Использует ReID кластеризацию для группировки обнаружений одного человека.",
            )
            if pro_settings["save_unique_people"]:
                pro_settings["unique_people_min_faces"] = st.slider(
                    "Мин. обнаружений лица",
                    min_value=1,
                    max_value=20,
                    value=3,
                    step=1,
                    help="Минимальное количество раз, когда человек должен быть обнаружен в видео. Фильтрует случайные появления и шум. Рекомендуется 3-5 для надежности.",
                )
                pro_settings["unique_people_quality_threshold"] = st.slider(
                    "Порог качества ReID",
                    min_value=0.5,
                    max_value=0.95,
                    value=0.7,
                    step=0.05,
                    help="Минимальная уверенность ReID для включения обнаружения в финальное фото. Более высокие значения (0.8+) дают точные, но менее полные траектории.",
                )

        # Performance settings
        with st.expander("Настройки производительности (PRO)", expanded=False):
            st.markdown("""
            **⚡ Оптимизации для длинных видео:**

            1. **Размер изображений LLaVA**: 224px = 2x быстрее, но ниже качество
            2. **Отключите детальное логирование** (чекбокс выше)
            3. **Уменьшите частоту кадров** в основных настройках
            4. **Используйте меньший batch_size** для LLaVA если память ограничена
            5. **Мониторьте использование VRAM**: nvidia-smi в терминале
            """)
            pro_settings["openai_batch_size"] = st.slider(
                "Размер батча OpenAI",
                min_value=1,
                max_value=32,
                value=8,
                step=1,
                help="Количество изображений, обрабатываемых параллельно в vLLM. Для GPU 16-80GB: 8-16. Для 2xGPU 160GB: 16-32. Большие значения ускоряют обработку, но требуют больше VRAM.",
            )
            pro_settings["yolo_batch_size"] = st.slider(
                "Размер батча YOLO",
                min_value=1,
                max_value=64,
                value=16,
                step=1,
                help="Количество изображений для одновременной обработки YOLO. Для 8 ядер CPU: 16-32. Для GPU: 8-16 достаточно. Влияет на скорость детекции объектов.",
            )
            pro_settings["llava_image_size"] = st.slider(
                "Размер изображений для LLaVA",
                min_value=224,
                max_value=672,
                value=336,
                step=112,
                help="Размер изображений для анализа LLaVA (в пикселях). ⚠️ Влияет на качество и скорость! 224px - 2x быстрее, но хуже качество. 336px - баланс. 448px+ - лучшее качество анализа, но медленнее.",
            )

        # Умная кластеризация видео
        with st.expander("Умная обработка длинных видео (PRO)", expanded=False):
            st.markdown("""
            **🎬 Умная кластеризация кадров:**

            Система анализирует видео по временным окнам, группирует похожие кадры
            по SSI схожести и выбирает репрезентативные кадры из каждого кластера.
            Уменьшает количество кадров для анализа без потери разнообразия сцен.
            """)

            pro_settings["enable_smart_clustering"] = st.checkbox(
                "Включить умную кластеризацию кадров",
                value=False,  # По умолчанию отключено
                help="Автоматическая оптимизация для длинных видео с низкой активностью. Значительно ускоряет анализ, но может пропустить некоторые детали.",
            )

            if pro_settings["enable_smart_clustering"]:
                pro_settings["clustering_ssi_threshold"] = st.slider(
                    "SSI порог схожести кадров",
                    min_value=0.7,
                    max_value=0.95,
                    value=0.85,
                    step=0.05,
                    help="Порог схожести для группировки похожих кадров. Выше = строже фильтрация, меньше кадров для анализа.",
                )

                pro_settings["clustering_window_duration"] = st.slider(
                    "Длительность временного окна (сек)",
                    min_value=60,
                    max_value=900,
                    value=300,
                    step=60,
                    help="Размер временного окна для кластеризации. Меньше = больше окон, дольше обработка, но выше точность.",
                )
            pro_settings["enable_detailed_logging"] = st.checkbox(
                "Детальное логирование",
                value=False,  # По умолчанию отключено для производительности
                help="Включить подробное логирование всех этапов обработки. Полезно для отладки, но замедляет работу на длинных видео.",
            )
            pro_settings["max_concurrent_frames"] = st.slider(
                "Максимум одновременных кадров",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Лимит кадров, обрабатываемых параллельно в конвейере. Для GPU 16-80GB: 30-80. Для 2xGPU: 80-150. Высокие значения ускоряют обработку длинных видео.",
            )

            # Advanced performance settings
            with st.expander("Расширенные настройки производительности ⚠️", expanded=False):
                pro_settings["enable_frame_deduplication"] = st.checkbox(
                    "❌ Дедупликация кадров (ОПАСНО!)",
                    value=False,  # Изменено с True на False по умолчанию
                    help="⚠️ ОПАСНО! Может пропустить важные кадры. Отключает дедупликацию кадров между разными видео. Использовать только если точно знаете что делаете!",
                )
                if pro_settings["enable_frame_deduplication"]:
                    st.error("⚠️ ВНИМАНИЕ: Дедупликация кадров может необратимо пропустить важные моменты!")
                    pro_settings["frame_similarity_threshold"] = st.slider(
                        "Порог схожести кадров ⚠️",
                        min_value=0.85,
                        max_value=0.99,
                        value=0.95,
                        step=0.01,
                        help="⚠️ ОПАСНО! Кадры с схожестью выше порога пропускаются. Может потерять важные изменения в видео. Использовать с крайней осторожностью!",
                    )
                    pro_settings["min_frames_between_duplicates"] = st.slider(
                        "Мин. интервал между дубликатами",
                        min_value=5,
                        max_value=60,
                        value=15,
                        step=5,
                        help="Минимальное количество кадров между похожими. Предотвращает слишком частые пропуски важных изменений. Для видео 30fps: 15 = минимум 0.5 секунды.",
                    )

                pro_settings["enable_adaptive_batch"] = st.checkbox(
                    "⚠ Адаптивный размер батча",
                    value=False,
                    help="❌ НЕ РЕАЛИЗОВАНО: Автоматическая подстройка размера батча под доступную GPU память. Пока используйте ручные настройки выше для каждого GPU отдельно.",
                )

        # Summarization settings
        with st.expander("Настройки суммаризации (PRO)", expanded=False):
            pro_settings["summarization_mode"] = st.selectbox(
                "Режим суммаризации",
                ["Стандартный (адаптивный)", "Детальный (больше чанков)", "Компактный (меньше чанков)", "Баланс (умолчание)"],
                index=3,
                help="Режим агрегации событий для LLM анализа. 'Стандартный' - автоматически подстраивает детализацию под длину видео. 'Детальный' - максимум информации. 'Компактный' - минимум данных. 'Баланс' - оптимальные настройки по умолчанию.",
            )
            if pro_settings["summarization_mode"] != "Баланс (умолчание)":
                pro_settings["custom_max_chunks"] = st.slider(
                    "Максимум временных чанков",
                    min_value=2,
                    max_value=20,
                    value=8,
                    step=1,
                    help="Максимальное количество временных интервалов (чанков) для группировки событий. Большие значения дают более детальный анализ, но дольше обрабатываются LLM.",
                )
                pro_settings["custom_max_evidence"] = st.slider(
                    "Максимум доказательств",
                    min_value=2,
                    max_value=15,
                    value=5,
                    step=1,
                    help="Максимальное количество конкретных событий (доказательств), передаваемых в LLM для анализа. Ограничивает контекст, чтобы избежать переполнения токенов.",
                )

    st.subheader("Модели")
    vision_llm_ui = st.radio(
        "Анализ кадров (куда уходит трафик изображений)",
        ["Local LLaVA", "GigaChat API", "OpenAI Local API", "Off (только CV)"],
        index=1,
    )
    final_llm_ui = st.radio(
        "Финальный ответ (текст/агрегация событий)",
        ["GigaChat API", "Local LLaVA (не рекомендуется)", "OpenAI Local API"],
        index=0,
    )

    gigachat_api_key = None
    gigachat_ca_cert_path = None
    need_gigachat = (vision_llm_ui == "GigaChat API") or (final_llm_ui == "GigaChat API")
    if need_gigachat:
        # По умолчанию берём ключ из config/gigachat_keys.json или env,
        # ручной ввод — только как override.
        default_key = load_gigachat_default_key()
        if default_key:
            gigachat_api_key = default_key
        else:
            # В продакшене ключ должен быть настроен заранее через env/config
            gigachat_api_key = None

        default_ca = load_gigachat_ca_bundle_file()
        gigachat_ca_cert_path = default_ca


# Выбор метода загрузки контента
content_input_method = st.radio(
    "Метод загрузки контента",
    ["Drag & Drop (Windows)", "Папка с файлами (Linux)", "Конкретный файл"],
    index=0,
    horizontal=True,
    help="Выберите удобный способ загрузки видео или изображений для вашей системы"
)

# Инициализация переменных
uploaded_files = []
content_paths_from_folder = []
single_content_path = ""

# Обработка загрузки в зависимости от выбранного метода
if content_input_method == "Drag & Drop (Windows)":
    uploaded_files = st.file_uploader(
        "Загрузите видео или изображения (макс. 2 ГБ на файл)",
        type=["mp4", "avi", "mov", "mkv", "wmv", "flv", "jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        help="Поддерживаемые форматы: видео (mp4, avi, mov, mkv, wmv, flv) и изображения (jpg, png, bmp, tiff, webp)"
    )

    if uploaded_files and len(uploaded_files) > 20:
        st.error("Можно загрузить максимум 20 файлов за раз.")
        uploaded_files = uploaded_files[:20]

elif content_input_method == "Папка с файлами (Linux)":
    folder_path = st.text_input(
        "Путь к папке с файлами",
        placeholder="/home/user/content/ или C:\\Users\\user\\content\\",
        help="Укажите полный путь к папке содержащей видео или изображения"
    )

    if folder_path:
        try:
            import os
            import glob

            # Проверяем существование папки
            if not os.path.exists(folder_path):
                st.error(f"Папка не найдена: {folder_path}")
            else:
                # Ищем все видео и изображения в папке
                video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
                found_files = []

                for ext in video_extensions + image_extensions:
                    pattern = os.path.join(folder_path, ext)
                    found_files.extend(glob.glob(pattern))

                if found_files:
                    content_paths_from_folder = sorted(found_files)
                    st.success(f"Найдено {len(content_paths_from_folder)} файлов в папке")

                    # Показываем найденные файлы
                    with st.expander("Найденные файлы", expanded=False):
                        for path in content_paths_from_folder:
                            file_type = "🎥 Видео" if any(path.lower().endswith(ext[1:]) for ext in video_extensions) else "🖼️ Изображение"
                            st.text(f"{file_type}: {os.path.basename(path)}")

                    # Показываем превью для первых нескольких файлов
                    if len(content_paths_from_folder) <= 10:  # Показываем превью только для небольшого количества
                        st.subheader("Превью файлов")
                        cols = st.columns(min(4, len(content_paths_from_folder)))
                        for i, file_path in enumerate(content_paths_from_folder):
                            with cols[i % len(cols)]:
                                if any(file_path.lower().endswith(ext[1:]) for ext in video_extensions):
                                    st.caption(f"🎥 {os.path.basename(file_path)}")
                                    try:
                                        st.video(file_path)
                                    except Exception as e:
                                        st.warning(f"Не удалось загрузить превью видео")
                                else:
                                    st.caption(f"🖼️ {os.path.basename(file_path)}")
                                    try:
                                        st.image(file_path, width=150)
                                    except Exception as e:
                                        st.warning(f"Не удалось загрузить превью изображения")
                else:
                    st.warning("В указанной папке не найдено поддерживаемых файлов (видео или изображения)")

        except Exception as e:
            st.error(f"Ошибка при сканировании папки: {str(e)}")

elif content_input_method == "Конкретный файл":
    single_content_path = st.text_input(
        "Путь к файлу",
        placeholder="/home/user/video.mp4 или /home/user/image.jpg",
        help="Укажите полный путь к конкретному видео файлу или изображению"
    )

    if single_content_path:
        import os
        if not os.path.exists(single_content_path):
            st.error(f"Файл не найден: {single_content_path}")
        else:
            # Проверяем расширение файла
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            _, ext = os.path.splitext(single_content_path.lower())

            if ext in video_extensions:
                st.success(f"🎥 Видео файл найден: {os.path.basename(single_content_path)}")
                # Показываем превью
                st.subheader("Превью видео")
                try:
                    st.video(single_content_path)
                except Exception as e:
                    st.warning(f"Не удалось загрузить превью файла")
            elif ext in image_extensions:
                st.success(f"🖼️ Изображение найдено: {os.path.basename(single_content_path)}")
                # Показываем превью
                st.subheader("Превью изображения")
                try:
                    st.image(single_content_path, caption=os.path.basename(single_content_path), width=400)
                except Exception as e:
                    st.warning(f"Не удалось загрузить превью файла")
            else:
                st.error(f"Неподдерживаемый формат файла. Поддерживаются видео ({', '.join(video_extensions)}) и изображения ({', '.join(image_extensions)})")
                single_content_path = ""
else:
    st.error("Не выбран способ загрузки контента")

# Обработка загруженных файлов для превью
if uploaded_files and content_input_method == "Drag & Drop (Windows)":
    st.subheader("Превью загруженных файлов")
    cols = st.columns(min(4, len(uploaded_files)))
    for i, f in enumerate(uploaded_files):
        with cols[i % len(cols)]:
            file_name = f.name
            file_ext = file_name.lower().split('.')[-1] if '.' in file_name else ''
            video_exts = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv']

            if file_ext in video_exts:
                st.caption(f"🎥 {file_name}")
                st.video(f)
            else:
                st.caption(f"🖼️ {file_name}")
                st.image(f, width=150)

user_query = st.text_area(
    "Ваш вопрос по видео",
    placeholder="Например: 'Что делают люди в кадре и как они перемещаются во времени?'",
)
require_json = st.checkbox("Требуется JSON-ответ", help="Строгий формат JSON для интеграций")

# Определяем, есть ли контент для анализа
has_content = (
    (uploaded_files and content_input_method == "Drag & Drop (Windows)") or
    (content_paths_from_folder and content_input_method == "Папка с файлами (Linux)") or
    (single_content_path and content_input_method == "Конкретный файл")
)

run_btn = st.button("Запустить анализ", type="primary", disabled=not has_content or not user_query)

if run_btn:
    # Получаем пути к файлам и определяем их типы
    video_paths = []
    image_paths = []

    if content_input_method == "Drag & Drop (Windows)":
        # Разделяем загруженные файлы на видео и изображения
        import tempfile
        temp_dir = tempfile.mkdtemp()

        for f in uploaded_files:
            file_name = f.name
            file_ext = file_name.lower().split('.')[-1] if '.' in file_name else ''
            video_exts = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv']

            file_path = os.path.join(temp_dir, file_name)
            with open(file_path, "wb") as file:
                file.write(f.getbuffer())

            if file_ext in video_exts:
                video_paths.append(file_path)
            else:
                image_paths.append(file_path)

    elif content_input_method == "Папка с файлами (Linux)":
        # Разделяем файлы из папки на видео и изображения
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

        for file_path in content_paths_from_folder:
            _, ext = os.path.splitext(file_path.lower())
            if ext in video_exts:
                video_paths.append(file_path)
            elif ext in image_exts:
                image_paths.append(file_path)

    elif content_input_method == "Конкретный файл":
        # Определяем тип одиночного файла
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

        _, ext = os.path.splitext(single_content_path.lower())
        if ext in video_exts:
            video_paths = [single_content_path]
        elif ext in image_exts:
            image_paths = [single_content_path]

    # Определяем режим работы
    is_image_mode = len(image_paths) > 0 and len(video_paths) == 0
    content_paths = video_paths + image_paths

    progress = st.progress(0)
    stage = st.empty()

    def _progress_cb(v: float, msg: str) -> None:
        progress.progress(int(v * 100))
        stage.info(msg)

    vision_llm = "llava_local" if vision_llm_ui == "Local LLaVA" else ("gigachat_api" if vision_llm_ui == "GigaChat API" else ("openai_local" if vision_llm_ui == "OpenAI Local API" else "off"))
    final_llm = "gigachat_api" if final_llm_ui == "GigaChat API" else ("llava_local" if final_llm_ui == "Local LLaVA (не рекомендуется)" else "openai_local")
    state: PipelineState = {
        "video_paths": video_paths,
        "user_query": user_query,
        "ui_mode": ui_mode,
        "pro_settings": pro_settings if ui_mode == "PRO" else {},
        "require_json": bool(require_json),
        "vision_llm": vision_llm,
        "final_llm": final_llm,
        "gigachat_api_key": gigachat_api_key,
        "gigachat_ca_cert_path": gigachat_ca_cert_path,
        "analyze_people": bool(analyze_people),
        "analyze_pose": bool(analyze_pose),
        "save_unique_people": bool(pro_settings.get("save_unique_people", False)),
        "unique_people_min_faces": int(pro_settings.get("unique_people_min_faces", 3)),
        "unique_people_quality_threshold": float(pro_settings.get("unique_people_quality_threshold", 0.7)),
        "summarization_mode": str(pro_settings.get("summarization_mode", "Баланс (умолчание)")),
        "custom_max_chunks": int(pro_settings.get("custom_max_chunks", 8)),
        "custom_max_evidence": int(pro_settings.get("custom_max_evidence", 5)),
        "openai_batch_size": int(pro_settings.get("openai_batch_size", 8)),
        "yolo_batch_size": int(pro_settings.get("yolo_batch_size", 16)),
        "max_concurrent_frames": int(pro_settings.get("max_concurrent_frames", 50)),
        "enable_frame_deduplication": bool(pro_settings.get("enable_frame_deduplication", False)),
        "frame_similarity_threshold": float(pro_settings.get("frame_similarity_threshold", 0.95)),
        "min_frames_between_duplicates": int(pro_settings.get("min_frames_between_duplicates", 15)),
        "enable_adaptive_batch": bool(pro_settings.get("enable_adaptive_batch", False)),
        "force_no_visual_analysis": bool(force_no_visual_analysis or vision_llm == "off"),
        "processing_log": [],
        "video_paths": video_paths,
        "image_paths": image_paths,
        "is_image_mode": is_image_mode,
    }

    t0 = time.time()
    with st.spinner("Выполняется анализ..."):
        out = run_pipeline(state, progress_cb=_progress_cb)
    elapsed = time.time() - t0

    st.success(f"Готово. Output: {out.get('result_path','')}")

    # Load events table
    out_dir = out.get("result_path") or out.get("output_dir")
    events_df = pd.DataFrame(out.get("events") or [])
    if out_dir and os.path.exists(os.path.join(out_dir, "events.parquet")):
        try:
            events_df = pd.read_parquet(os.path.join(out_dir, "events.parquet"))
        except Exception:
            pass

    tabs = st.tabs(["Ответ LLM", "События", "Логи обработки"])

    with tabs[0]:
        final_answer = out.get("final_answer")
        if require_json and isinstance(final_answer, dict):
            st.json(final_answer)
        else:
            st.markdown(str(final_answer or ""))
        models_used = out.get("models_used") or []
        st.caption(f"Обработано за {elapsed:.1f} сек. Использовано моделей: {', '.join(models_used) or 'n/a'}")

    with tabs[1]:
        if events_df.empty:
            st.info("События отсутствуют.")
        else:
            st.dataframe(
                events_df,
                width="stretch",
                hide_index=True,
            )

    with tabs[2]:
        log_text = "\n".join(out.get("processing_log") or [])
        if out_dir and os.path.exists(os.path.join(out_dir, "processing_log.log")):
            try:
                with open(os.path.join(out_dir, "processing_log.log"), "r", encoding="utf-8") as f:
                    log_text = f.read()
            except Exception:
                pass
        st.text_area("processing_log.log", value=log_text, height=320)

    if out.get("result_zip_bytes"):
        st.download_button(
            "Скачать результаты (ZIP)",
            data=out["result_zip_bytes"],
            file_name=os.path.basename(out_dir or "results") + ".zip",
            mime="application/zip",
        )


