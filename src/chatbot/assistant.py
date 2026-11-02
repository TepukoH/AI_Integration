import os
from pathlib import Path

from .knowledge_base import KnowledgeBase

DEFAULT_FAQ = Path(__file__).resolve().parent.parent / "data" / "faq.json"

SYSTEM_PROMPT = (
    "Ты — вежливый ассистент Московского финансово-промышленного университета "
    "«Синергия». Отвечай кратко, по делу, опираясь ТОЛЬКО на предоставленный "
    "контекст. Если в контексте нет ответа — попроси уточнить у приёмной комиссии "
    "(8 800 100-00-11) или через личный кабинет."
)


class SynergyAssistant:
    def __init__(self, faq_path: str | Path = DEFAULT_FAQ, top_k: int = 3,
                 openai_key: str | None = None, model: str | None = None):
        self.kb = KnowledgeBase(faq_path)
        self.top_k = top_k
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = None
        if self.openai_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.openai_key)
            except ImportError:
                self._client = None

    @property
    def llm_enabled(self) -> bool:
        return self._client is not None

    def answer(self, question: str, use_llm: bool | None = None) -> dict:
        candidates = self.kb.search(question, top_k=self.top_k)
        best = candidates[0] if candidates else None

        if use_llm is None:
            use_llm = self.llm_enabled

        if use_llm and self._client and candidates:
            context = "\n\n".join(
                f"Вопрос: {c['question']}\nОтвет: {c['answer']}"
                for c in candidates
            )
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",
                         "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
                    ],
                )
                return {
                    "answer": resp.choices[0].message.content.strip(),
                    "source": "llm+rag",
                    "confidence": best["score"] if best else 0.0,
                    "retrieved": candidates,
                }
            except Exception:
                pass  # молча откатываемся в FAQ-фолбэк

        if best:
            return {
                "answer": best["answer"],
                "source": "faq",
                "confidence": best["score"],
                "retrieved": candidates,
            }
        return {
            "answer": ("К сожалению, точного ответа не нашлось. "
                       "Пожалуйста, свяжитесь с приёмной комиссией: "
                       "8 (800) 100-00-11 или через личный кабинет."),
            "source": "fallback",
            "confidence": 0.0,
            "retrieved": [],
        }