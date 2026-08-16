"""Gradio UI.  python app.py -> http://localhost:7860"""

import gradio as gr

from eulaw import config


def format_sources(sources: list[dict]) -> str:
    """Markdown list, each source deep-linked to its provision on EUR-Lex."""
    from eulaw.retrieve import cite

    if not sources:
        return ""
    lines = ["**Sources**", ""]
    for i, s in enumerate(sources, start=1):
        lines.append(f"{i}. [{cite(s)}]({s['url']}) — score {s['score']:.2f}")
    return "\n".join(lines)


def ask(question: str, k: int) -> tuple[str, str]:
    from eulaw.generate import answer  # lazy: UI starts before models load

    if not question.strip():
        return "Please enter a question.", ""
    try:
        result = answer(question, k=int(k))
    except Exception as e:
        return (f"**The language model is unavailable** ({type(e).__name__}). "
                "Check that the LLM server is running and try again.", "")
    return result["answer"], format_sources(result["sources"])


with gr.Blocks(title=config.TITLE) as demo:
    gr.Markdown(
        f"# {config.TITLE}\n{config.DESCRIPTION}\n\n"
        "Answers are grounded in these regulations and cited; if they do not "
        "contain the answer, the system says so instead of guessing.\n\n"
        f"> {config.DISCLAIMER}"
    )
    with gr.Row():
        question = gr.Textbox(label="Your question", scale=4,
                              placeholder="e.g. When is an AI system classified as high-risk?")
        k = gr.Slider(3, 10, value=config.TOP_K, step=1, scale=1,
                      label="Retrieved provisions (k)")
    ask_btn = gr.Button("Ask", variant="primary")
    answer_md = gr.Markdown()
    sources_md = gr.Markdown()
    gr.Examples([[q] for q in config.EXAMPLES], inputs=question)

    ask_btn.click(ask, inputs=[question, k], outputs=[answer_md, sources_md])
    question.submit(ask, inputs=[question, k], outputs=[answer_md, sources_md])


if __name__ == "__main__":
    demo.launch()
