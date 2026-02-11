import gradio as gr
from database import VectorDatabase
from Query_data import query_rag

print("Loading database and model... please wait.")
db_instance = VectorDatabase() 

def rag_chatbot(user_query, history_state):
    if not user_query.strip():
        return "Please enter a question.", "No sources", history_state

    memory_text = ""
    for user_msg, bot_msg in history_state[-5:]:
        memory_text += f"User: {user_msg}\nBot: {bot_msg}\n"

    answer_text, sources_list = query_rag(user_query, db_instance, memory_text)
    
    history_state.append((user_query, answer_text))
    
    sources_text = ", ".join(sources_list) if sources_list else "SQL Database / General Knowledge"
    return answer_text, sources_text, history_state

with gr.Blocks(title="RAG Knowledge Assistant") as demo:
    history_state = gr.State([]) 
    
    gr.Markdown("#Multi-Source RAG Chatbot")
    gr.Markdown("Query your **SQL Database** and **Document Store** (PDF/TXT) in one place.")

    with gr.Row():
        with gr.Column(scale=4):
            question = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., How many employees are in the database?", 
                lines=2
            )
            ask_btn = gr.Button("Submit Question", variant="primary")
        
    answer = gr.Textbox(label="Bot's Answer", lines=10, interactive=False)
    sources = gr.Textbox(label="Sources Consulted", lines=2, interactive=False)

    ask_btn.click(
        fn=rag_chatbot,
        inputs=[question, history_state],
        outputs=[answer, sources, history_state]
    )

demo.launch()