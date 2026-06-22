from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_API_KEY = "" # for local model usage, api key isn't needed

def build_chat_model() -> ChatOllama:
    """
    Creates a ChatOllama 
    """

    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )

    return model

def build_chain():
    """
    Builds a simple Chain using LCEL.

    ChatPromptTemplate -> ChatOllama => StrOutputParser
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a beginner friendly learning assistant for our students.

                Rules:
                1. Explain the concept clearly and in a beginner friendly way.
                2. Give answer in bullet points.
                """
            ),
            (
                "human",
                "Explain me {topic} in easy words"
            )
        ]
    )

    model = build_chat_model()

    parser = StrOutputParser()

    #build the chain

    chain = prompt | model | parser

    return chain

chain = build_chain()

response = chain.invoke({
    "topic" : "Agentic AI"
})

print(response)
