"""Reranker service layer for post-FAISS chunk scoring."""

from abc import ABC, abstractmethod

from src.models import RetrievedChunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


class IdentityReranker(BaseReranker):
    """No-op — returns first top_k chunks unchanged (RERANKER_TYPE=none)."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]


class CrossEncoderReranker(BaseReranker):
    """Local cross-encoder via sentence-transformers. Lazy-loads model on first call."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        texts = [chunk.text or chunk.content for chunk in chunks]
        scores = self._model.predict([(query, t) for t in texts])
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [
            chunk.model_copy(update={"rerank_score": float(score)})
            for score, chunk in ranked[:top_k]
        ]


class LLMReranker(BaseReranker):
    """LLM-based reranker — zero new dependencies; uses existing OpenRouter chat model.

    Trade-off: 1 LLM call per chunk → higher latency and token cost.
    """

    _PROMPT = (
        "Rate the relevance of the document to the query on a scale of 0-10.\n"
        "Reply with ONLY a single integer.\n\nQuery: {query}\n\nDocument:\n{doc}"
    )

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        from src.llm.client import get_chat_model
        model = get_chat_model()
        scored = []
        for chunk in chunks:
            text = (chunk.text or chunk.content)[:800]
            prompt = self._PROMPT.format(query=query, doc=text)
            try:
                resp = model.invoke([{"role": "user", "content": prompt}])
                score = float(resp.content.strip())
            except Exception:
                score = 0.0
            scored.append((score, chunk))
        return [
            c.model_copy(update={"rerank_score": s})
            for s, c in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
        ]


class CohereReranker(BaseReranker):
    """Cohere Rerank API — requires cohere>=5.0.0 and COHERE_API_KEY."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if self._client is None:
            import cohere
            self._client = cohere.Client(api_key=self._api_key)
        docs = [chunk.text or chunk.content for chunk in chunks]
        results = self._client.rerank(model=self._model, query=query, documents=docs, top_n=top_k)
        return [
            chunks[r.index].model_copy(update={"rerank_score": float(r.relevance_score)})
            for r in results.results
        ]


def get_reranker() -> BaseReranker:
    from src.config import settings
    t = settings.reranker_type
    if t == "cross_encoder":
        return CrossEncoderReranker(settings.cross_encoder_model)
    if t == "llm":
        return LLMReranker()
    if t == "cohere":
        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY must be set when RERANKER_TYPE=cohere")
        return CohereReranker(settings.cohere_api_key, settings.cohere_rerank_model)
    return IdentityReranker()


reranker = get_reranker()
