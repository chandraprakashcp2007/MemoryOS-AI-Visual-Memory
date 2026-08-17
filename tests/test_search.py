from backend.services.search_service import SearchQuery, SearchService


class DummyMemoryProvider:
    def get_memory(self, memory_id):
        return {
            "memory_id": memory_id,
            "title": "Recipe screenshot",
            "summary": "Food recipe discovered in screenshot",
            "category": "recipe",
            "intent": "general",
            "image_url": "/upload/file/mem_abc123",
            "preview_url": "/upload/file/mem_abc123",
            "thumbnail_url": "/upload/file/mem_abc123",
            "metadata": {
                "image_url": "/upload/file/mem_abc123",
                "preview_url": "/upload/file/mem_abc123",
                "thumbnail_url": "/upload/file/mem_abc123",
            },
        }


def test_build_results_preserve_original_image_reference():
    service = SearchService(memory_provider=DummyMemoryProvider())

    results = service.build_results(
        query=SearchQuery(text="food", top_k=5),
        candidates={
            "mem_abc123": {
                "semantic_score": 0.95,
                "keyword_score": 0.50,
                "metadata_score": 0.0,
                "metadata": {},
            }
        },
    )

    assert len(results) == 1
    assert results[0].memory_id == "mem_abc123"
    assert results[0].image_url == "/upload/file/mem_abc123"
    assert results[0].preview_url == "/upload/file/mem_abc123"
    assert results[0].thumbnail_url == "/upload/file/mem_abc123"
