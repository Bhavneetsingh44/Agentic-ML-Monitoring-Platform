from src.agents.llm_reasoner import get_llm

llm = get_llm()

print("Model being used:", llm.model_name)

response = llm.invoke(
    "Explain data drift in one simple sentence."
)

print("LLM Response:", response.content)