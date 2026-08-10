"""Gradio web UI: ask a question, get a grounded answer with cited sources.

Run:  python app.py   ->  http://localhost:7860
"""

import gradio as gr

from askarxiv import config

EXAMPLES = [
    "What methods make LLM inference more efficient?",
    "What is Salience Bias in commonsense reasoning?",
    "How does ParliamentBench evaluate deception in LLM agents?",
    "What is the capital of Austria?",   # demo of the refusal contract
]


def format_sources(sources: list[dict]) -> str:
    """Render retrieved chunks as a Markdown list with arXiv links."""
    if not sources:
        return ""
    lines = ["**Sources**", ""]
    for i, s in enumerate(sources, start=1):
        url = f"https://arxiv.org/abs/{s['paper_id']}"
        lines.append(f"{i}. [{s['title']}]({url}) — chunk {s['chunk_index']}, "
                     f"score {s['score']:.2f}")
    return "\n".join(lines)


def ask(question: str, k: int) -> tuple[str, str]:
    """Gradio callback: run the full RAG loop for one question."""
    from askarxiv.generate import answer  # lazy: UI starts before models load

    if not question.strip():
        return "Please enter a question.", ""
    try:
        result = answer(question, k=int(k))
    except Exception as e:
        return (f"**The language model is unavailable** ({type(e).__name__}). "
                "Check that the LLM server is running and try again.", "")
    return result["answer"], format_sources(result["sources"])


with gr.Blocks(title="AskArxiv") as demo:
    gr.Markdown(
        "# AskArxiv\n"
        "Ask questions about 50 recent LLM papers (arXiv, cs.CL). Answers are "
        "grounded in the papers and cited; if the papers don't contain the "
        "answer, the system says so instead of guessing."
    )
    with gr.Row():
        question = gr.Textbox(label="Your question", scale=4,
                              placeholder="e.g. What methods make LLM inference more efficient?")
        k = gr.Slider(3, 10, value=config.TOP_K, step=1, scale=1,
                      label="Retrieved chunks (k)")
    ask_btn = gr.Button("Ask", variant="primary")
    answer_md = gr.Markdown()
    sources_md = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=question)

    ask_btn.click(ask, inputs=[question, k], outputs=[answer_md, sources_md])
    question.submit(ask, inputs=[question, k], outputs=[answer_md, sources_md])


if __name__ == "__main__":
    demo.launch()
