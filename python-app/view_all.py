from qdrant_client import QdrantClient
import json

client = QdrantClient(host="localhost", port=6333)

print("="*70)
print("📄 ВСЕ ДОКУМЕНТЫ В КОЛЛЕКЦИИ 'rag_documents':")
print("="*70)

# scroll возвращает кортеж: (список_точек, next_page_offset)
points, next_offset = client.scroll(
    collection_name="rag_documents",
    limit=100,
    with_payload=True,
    with_vectors=False
)

print(f"\nНайдено точек: {len(points)}\n")

for point in points:
    print(f"🆔 ID: {point.id}")
    print(f"📂 Источник: {point.payload.get('source', 'не указан')}")
    print(f"📝 Текст:")
    print(f"   {point.payload.get('text', 'Нет текста')}")
    print("-"*70)