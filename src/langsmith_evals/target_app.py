"""The application under evaluation: a minimal QA chain.

Swap this out for your real chain/agent — the eval scripts only depend on
`answer_question(question: str) -> str`.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from langsmith_evals.providers import get_chat_model

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise, accurate assistant. Answer in at most 3 sentences."),
        ("human", "{question}"),
    ]
)


def build_chain(provider: str | None = None, model: str | None = None):
    llm = get_chat_model(provider=provider, model=model)
    return _PROMPT | llm


def answer_question(question: str, provider: str | None = None, model: str | None = None) -> str:
    chain = build_chain(provider=provider, model=model)
    return chain.invoke({"question": question}).content
