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

Вот отформатированный текст, готовый для вставки в `README.md` на GitHub. Я использовал стандартные элементы Markdown: заголовки, списки, блоки кода и цитаты, чтобы структура была четкой и читаемой.

***

# Local RAG Gemma 4

## 📥 Установка

### 1. Клонирование репозитория
```bash
git clone https://github.com/Kord119/local-rag-gemma4.git
cd local-rag-gemma4
```

### 2. Подготовка моделей
Скачайте модели **Gemma 4** в папку `models/`:

*   `gemma-4-12B-it-QAT-Q4_0.gguf` (~6.5 GB) — основная LLM
*   `mmproj-gemma-4-12B-it-QAT-BF16.gguf` (~168 MB) — vision encoder для мультимодальности

> **Примечание:** Модели не включены в репозиторий из-за ограничений GitHub на размер файлов. Скачайте их с [Hugging Face](https://huggingface.com).

### 3. Запуск сервисов
Запустите все контейнеры:
```bash
docker compose up -d
```

Проверьте логи загрузки модели:
```bash
docker logs llamacpp -f
```
Дождитесь появления строки `llama server listening on http://0.0.0.0:8000` (обычно 1-3 минуты).

Настройте Python-окружение:
```bash
cd python-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Использование

### Добавление документов

**Способ 1: Загрузка из папки**  
Поместите `.txt` файлы в папку `documents/` и выполните:
```bash
python producer.py --load
```

**Способ 2: В интерактивном режиме**  
Запустите скрипт:
```bash
python producer.py
```
В интерактивном режиме доступна команда `/add`:
> `/add Текст документа --source=название_источника`

### Запросы к системе

*   **В интерактивном режиме:** Вводите вопрос после приглашения `❓ Ваш вопрос:`.
*   **Из командной строки:**
    ```bash
    python producer.py "Для чего Docker использует контейнеры?"
    ```

### Веб-интерфейсы
После запуска сервисов доступны следующие интерфейсы:

| Сервис | Адрес |
| :--- | :--- |
| **LLM Chat** | `http://localhost:8000` |
| **Qdrant Dashboard** | `http://localhost:6333/dashboard` |
| **RabbitMQ Management** | `http://localhost:15672` (логин/пароль: `guest/guest`) |

---

## ⚙️ Архитектура и распределение нагрузки

*   **LLM (Gemma 4) → GPU:** Генерация текста требует массового параллелизма.
*   **Embeddings (BGE-M3) → CPU:** Экономит VRAM, обеспечивает быстрый инференс.
*   **Vector DB (Qdrant) → RAM/SSD:** Осуществляет векторный поиск.

---

## 🛠 Технические решения

1.  **QAT-квантование:** Gemma 4 QAT-Q4 помещается в 16GB VRAM с запасом для KV-cache, сохраняя **95%+ качества** по сравнению с FP16.
2.  **Multilingual embeddings:** BGE-M3 от BAAI обеспечивает качественный кросс-лингвальный поиск (русский ↔ английский, транслитерация).
3.  **Thinking Mode:** Корректная обработка поля `reasoning_content` в ответах Gemma 4 с fallback при пустом `content`.

---

## ⚠️ Известные проблемы и решения

*   **Ошибка OOM (Out Of Memory):** Если модель не загружается, уменьшите параметр `MAX_TOKENS` в `producer.py` или используйте меньшую контекстную длину.
*   **Docker не виден в WSL:** Включите интеграцию в Docker Desktop:  
    `Settings → Resources → WSL Integration → Ubuntu: ON`.
*   **Медленная загрузка embeddings:** BGE-M3 (2.2 ГБ) загружается ~30 сек при первом запуске.  
    *Решение:* Для ускорения используйте модель `BAAI/bge-small-ru` (130 МБ), изменив переменную `EMBEDDING_MODEL` в `producer.py`.

---

## 📄 License
MIT License - см. файл [LICENSE](LICENSE).
