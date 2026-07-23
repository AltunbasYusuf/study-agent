from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from tools import ask_pdf, save_summary

MODEL = "qwen2.5:7b"

llm = ChatOllama(model=MODEL, temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[ask_pdf, save_summary],
    prompt="When reporting a tool's result to the user (such as a file path), "
           "quote it EXACTLY as returned by the tool. Never modify, complete, "
           "or guess at filenames or paths yourself.",
)


def run_agent(user_question: str):
    print(f'User: "{user_question}"\n')

    result = agent.invoke({
        "messages": [{"role": "user", "content": user_question}]
    })

    for msg in result["messages"]:
        role = msg.type
        if role == "tool":
            print(f"[tool result] {msg.content[:150]}...")
        elif role == "ai" and msg.content:
            print(f"[agent] {msg.content}")
        elif role == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"[agent decides] call {tc['name']} with {tc['args']}")

    print("\n--- FINAL ANSWER ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    run_agent("Can you make me a study summary of the different testing strategies (top-down, bottom-up, thread, stress)?")