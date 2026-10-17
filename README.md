# Local RAG System с Gemma 4 12B

Полностью локальная RAG-система (Retrieval-Augmented Generation) на базе Gemma 4 12B с GPU-ускорением. Все компоненты работают оффлайн, данные не покидают устройство.

## Возможности

- Локальная LLM Gemma 4 12B с QAT-Q4 квантованием
- GPU-ускоренный инференс через llama.cpp с CUDA
- Семантический поиск на базе BGE-M3 embeddings с поддержкой 100+ языков
- Векторная база данных Qdrant для хранения эмбеддингов
- Мультимодальность (работа с изображениями через mmproj)
- Поддержка thinking mode Gemma 4
- Развёртывание через Docker Compose

## Требования

- **ОС**: Windows 11 с WSL2 (Ubuntu 24.04) или Linux
- **GPU**: NVIDIA с 16+ GB VRAM (протестировано на RTX 4060 Ti 16GB)
- **RAM**: 16+ GB
- **Диск**: 20+ GB свободного места
- **Драйвер**: NVIDIA Windows Driver 535+ с поддержкой WSL2 CUDA
- **Docker Desktop** с включённой WSL Integration

## Установка

### 1. Клонирование репозитория

git clone https://github.com/Kord119/local-rag-gemma4.git
cd local-rag-gemma4
### 2. Подготовка моделей
Скачайте модели Gemma 4 в папку models/:
gemma-4-12B-it-QAT-Q4_0.gguf (~6.5 GB) — основная LLM
mmproj-gemma-4-12B-it-QAT-BF16.gguf (~168 MB) — vision encoder для мультимодальности
Модели не включены в репозиторий из-за ограничений GitHub на размер файлов.
### 3. Запуск сервисов
1 docker compose up -d
Проверьте логи загрузки модели:
docker logs llamacpp -f
Дождитесь появления строки llama server listening on http://0.0.0.0:8000 (обычно 1-3 минуты).
cd python-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Использование
Добавление документов
Способ 1: Загрузка из папки
Поместите .txt файлы в папку documents/ и выполните:
python producer.py --load
Способ 2: В интерактивном режиме
python producer.py
В интерактивном режиме доступна команда:
/add Текст документа --source=название_источника
### Запросы к системе
В интерактивном режиме: ❓ Ваш вопрос:
Или из командной строки:
python producer.py "Для чего Docker использует контейнеры?"
Веб-интерфейсы
После запуска сервисов доступны:
LLM Chat: http://localhost:8000
Qdrant Dashboard: http://localhost:6333/dashboard
RabbitMQ Management: http://localhost:15672 (guest/guest)
Архитектура
┌─────────────────────────────────────────────┐
│              Docker Compose                  │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │llama.cpp │  │ Qdrant   │  │ RabbitMQ │  │
│  │:8000     │  │ :6333    │  │ :5672    │  │
│  │  GPU     │  │  Vector  │  │  Queue   │  │
│  └────┬─────┘  └────┬─────┘  └──────────┘  │
└───────┼─────────────┼────────────────────────┘
        │             │
        ▼             ▼
┌─────────────────────────────────────────────┐
│      Python RAG App                          │
│      - BGE-M3 embeddings (CPU)               │
│      - Sentence Transformers                 │
└─────────────────────────────────────────────┘
### Распределение нагрузки:
LLM (Gemma 4) → GPU: генерация текста требует массового параллелизма
Embeddings (BGE-M3) → CPU: экономит VRAM, быстрый инференс
Vector DB (Qdrant) → RAM/SSD: векторный поиск

### Технические решения
QAT-квантование: Gemma 4 QAT-Q4 помещается в 16GB VRAM с запасом для KV-cache, сохраняя 95%+ качества FP16.
Multilingual embeddings: BGE-M3 от BAAI обеспечивает качественный кросс-лингвальный поиск (русский ↔ английский, транслитерация).
Thinking Mode: Корректная обработка поля reasoning_content в ответах Gemma 4 с fallback при пустом content.
### Известные проблемы
Если модель не загружается (OOM), то уменьшите MAX_TOKENS в producer.py или используйте меньшую контекстную длину.
Если Docker не виден в WSL то включите WSL Integration в Docker Desktop:
Settings → Resources → WSL Integration → Ubuntu: ON
Если происходит медленная загрузка embeddings, BGE-M3 (2.2 ГБ) загружается ~30 сек при первом запуске. Для ускорения используйте BAAI/bge-small-ru (130 МБ) — измените EMBEDDING_MODEL в producer.py.
License
MIT License - см. файл LICENSE
