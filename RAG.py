from database import VectorDatabase
from Query_data import query_rag

print("Loading database and model... please wait.")
db_instance = VectorDatabase()

print("--- RAG Chatbot Initialized ---")
conversation_memory = []

while True:
    user_query = input("\nAsk a question (or type 'quit' to exit): ")
    if user_query.lower() in ['quit', 'exit', 'q']:
        break

    memory_text = ""
    for h in conversation_memory[-6:]:
        memory_text += f"User: {h['user']}\nBot: {h['bot']}\n"

    print("Searching...")

    answer, sources_list = query_rag(
        user_query,
        db_instance,
        memory_text
    )

    conversation_memory.append({
        "user": user_query,
        "bot": answer
    })

    print("\n" + "=" * 60)
    print("BOT'S ANSWER:\n", answer)
    print("-" * 60)

    if sources_list:
        print("SOURCES CONSULTED:", ", ".join(sources_list))

    print("=" * 60)
