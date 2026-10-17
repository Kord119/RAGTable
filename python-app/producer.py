"""
RAG Producer - интеллектуальный ассистент на базе Gemma 4 + Qdrant
"""

import requests
import json
import sys
from typing import List, Dict, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

LLM_URL = "http://localhost:8000/v1/chat/completions"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "gemma"
MAX_TOKENS = 6000
TEMPERATURE = 0.7
TOP_K = 5
SCORE_THRESHOLD = 0.3

# ============================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================

print("🔄 Загрузка embedding модели на CPU...")
encoder = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
print("✅ Embedding модель готова")


def safe_encode(text) -> List[float]:
    """Безопасное кодирование текста с проверкой типа"""
    if text is None:
        raise ValueError("Текст для кодирования не может быть None")
    
    if not isinstance(text, str):
        text = str(text)
    
    text = text.strip()
    if not text:
        raise ValueError("Текст для кодирования пустой")
    
    return encoder.encode(text).tolist()


# ============================================
# ПРОВЕРКА СЕРВИСОВ
# ============================================

def check_services() -> bool:
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        if r.status_code != 200:
            print("❌ LLM не отвечает")
            return False
    except Exception as e:
        print(f"❌ LLM недоступен: {e}")
        return False
    
    try:
        r = requests.get(f"{QDRANT_URL}/collections", timeout=5)
        if r.status_code != 200:
            print("❌ Qdrant не отвечает")
            return False
    except Exception as e:
        print(f"❌ Qdrant недоступен: {e}")
        return False
    
    print("✅ Все сервисы доступны")
    return True


def ensure_collection_exists():
    try:
        r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        if r.status_code == 200:
            return
        
        requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
            json={
                "vectors": {
                    "size": 1024,
                    "distance": "Cosine"
                }
            }
        )
        print(f"✅ Коллекция '{COLLECTION_NAME}' создана")
    except Exception as e:
        print(f"⚠️  Ошибка: {e}")


# ============================================
# QDRANT
# ============================================

def search_documents(query: str, top_k: int = TOP_K) -> List[Dict]:
    """Ищет релевантные документы в Qdrant"""
    try:
        # БЕЗОПАСНОЕ кодирование запроса
        query_vector = safe_encode(query)
        
        response = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
            json={
                "vector": query_vector,
                "limit": top_k,
                "with_payload": True,
                "score_threshold": SCORE_THRESHOLD
            }
        )
        
        if response.status_code != 200:
            print(f"⚠️  Qdrant: {response.text}")
            return []
        
        return response.json().get("result", [])
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return []


def add_document(text: str, source: str, doc_id: Optional[int] = None) -> int:
    """Добавляет документ в Qdrant"""
    import uuid
    
    # Проверка входных данных
    if not text or not isinstance(text, str):
        print(f"❌ Невалидный текст документа")
        return -1
    
    if doc_id is None:
        doc_id = int(uuid.uuid4().int % (10**9))
    
    try:
        vector = safe_encode(text)
        
        response = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
            json={
                "points": [{
                    "id": doc_id,
                    "vector": vector,
                    "payload": {
                        "text": text,
                        "source": source,
                        "added_at": datetime.now().isoformat()
                    }
                }]
            }
        )
        
        if response.status_code in [200, 201]:
            print(f"✅ Документ добавлен (ID={doc_id})")
            return doc_id
        else:
            print(f"❌ Ошибка: {response.text}")
            return -1
    except Exception as e:
        print(f"❌ Ошибка добавления: {e}")
        return -1


# ============================================
# LLM
# ============================================

def build_prompt(question: str, context_docs: List[Dict]) -> str:
    """Формирует промпт с контекстом"""
    if not context_docs:
        return f"""Вопрос: {question}

Ответь на вопрос, используя свои общие знания. Если не знаешь точного ответа, честно об этом скажи."""
    
    # ИСПРАВЛЕНО: правильный доступ к payload
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        # В Qdrant данные лежат в doc["payload"]!
        payload = doc.get("payload", {})
        text = payload.get("text", "")
        source = payload.get("source", "unknown")
        score = doc.get("score", 0)
        
        if text:  # Только если текст не пустой
            context_parts.append(f"[{i}] (источник: {source}, релевантность: {score:.2f})\n{text}")
    
    if not context_parts:
        return f"""Вопрос: {question}

Ответь на вопрос, используя свои общие знания."""
    
    context_text = "\n\n".join(context_parts)
    
    return f"""Ты — интеллектуальный ассистент. Используй контекст для ответа.

КОНТЕКСТ:
{context_text}

ВОПРОС: {question}

ИНСТРУКЦИИ:
- Отвечай ТОЛЬКО на основе контекста
- Если ответа нет в контексте, скажи: "В документах нет информации"
- Цитируй источники: [1], [2]
- Будь кратким

ОТВЕТ:"""


def ask_llm(prompt: str) -> str:
    try:
        response = requests.post(
            LLM_URL,
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"❌ LLM error ({response.status_code}): {response.text}"
    
    except requests.exceptions.Timeout:
        return "❌ Timeout от LLM"
    except Exception as e:
        return f"❌ Ошибка LLM: {e}"


# ============================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================

def rag_query(question: str, verbose: bool = True) -> str:
    if verbose:
        print(f"\n{'='*60}")
        print(f"❓ Вопрос: {question}")
        print(f"{'='*60}")
    
    # 1. Поиск
    if verbose:
        print("\n🔍 Поиск в базе знаний...")
    
    context_docs = search_documents(question)
    
    if verbose:
        if context_docs:
            print(f"✅ Найдено {len(context_docs)} документов")
            for i, doc in enumerate(context_docs, 1):
                payload = doc.get("payload", {})
                source = payload.get("source", "unknown")
                score = doc.get("score", 0)
                print(f"   [{i}] {source} (score: {score:.2f})")
        else:
            print("⚠️  Документов не найдено, отвечаю из общих знаний")
    
    # 2. Промпт
    prompt = build_prompt(question, context_docs)
    
    # 3. LLM
    if verbose:
        print("\n🤖 Генерация ответа...")
    
    answer = ask_llm(prompt)
    
    if verbose:
        print(f"\n{'='*60}")
        print("💬 Ответ:")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")
    
    return answer


# ============================================
# ИНТЕРАКТИВНЫЙ РЕЖИМ
# ============================================

def interactive_mode():
    print("\n" + "="*60)
    print("🤖 RAG-АССИСТЕНТ НА БАЗЕ GEMMA 4")
    print("="*60)
    print("Команды:")
    print("  /add <текст> --source=<источник>  - добавить документ")
    print("  /stats                             - статистика")
    print("  /quit или /exit                    - выход")
    print("="*60 + "\n")
    
    while True:
        try:
            question = input("❓ Ваш вопрос: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["/quit", "/exit", "exit", "quit"]:
                print("👋 До свидания!")
                break
            
            # Команда добавления
            if question.startswith("/add"):
                parts = question.split("--source=")
                if len(parts) == 2:
                    text = parts[0].replace("/add", "").strip()
                    source = parts[1].strip()
                    if text:
                        add_document(text, source)
                    else:
                        print("❌ Текст документа пустой")
                else:
                    print("Использование: /add текст документа --source=источник")
                continue
            
            # Статистика
            if question == "/stats":
                try:
                    r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
                    info = r.json().get("result", {})
                    print(f"📊 Документов: {info.get('points_count', 0)}")
                except Exception as e:
                    print(f"❌ Ошибка: {e}")
                continue
            
            # RAG-запрос
            rag_query(question)
        
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()


def load_documents_from_folder(folder_path: str = "/mnt/o/Project/documents"):
    from pathlib import Path
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Папка {folder_path} не найдена")
        return
    
    txt_files = list(folder.glob("*.txt"))
    print(f"📂 Найдено {len(txt_files)} файлов")
    
    for file_path in txt_files:
        try:
            text = file_path.read_text(encoding='utf-8')
            source = file_path.stem
            if text.strip():
                add_document(text, source)
        except Exception as e:
            print(f"❌ Ошибка {file_path.name}: {e}")


if __name__ == "__main__":
    if not check_services():
        sys.exit(1)
    
    ensure_collection_exists()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--load":
        load_documents_from_folder()
        sys.exit(0)
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        rag_query(question)
        sys.exit(0)
    
    interactive_mode()