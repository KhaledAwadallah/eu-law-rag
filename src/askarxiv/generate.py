"""Step 5: build the prompt from retrieved chunks and call the LLM.

Will provide:
    answer(question, chunks, llm) -> str   # grounded answer with citations
    get_llm() -> callable                  # provider-agnostic LLM client
"""
