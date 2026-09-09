from typing import Any, Dict, List
from generation.config import MAX_CONTEXT_CHARS


def build_prompt(query: str, chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    context_blocks = []
    current_length = 0

    for chunk in chunks:
        text = (chunk.get("chunk_text") or chunk.get("text") or "").strip()
        if not text:
            continue

        if current_length + len(text) > max_chars and context_blocks:
            remaining = max_chars - current_length
            if remaining > 100:
                context_blocks.append(text[:remaining] + "...")
            break

        context_blocks.append(text)
        current_length += len(text)

    joined_context = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

    prompt = (
        "You are a helpful assistant for National Institute of Technology Agartala.\n\n"
        "Answer the question using only the supplied context.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n\n{joined_context}\n\n"
        "Answer:"
    )

    return prompt
