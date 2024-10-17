from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model_id = "phi3"

llm = ChatOllama(
    model=model_id,
    model_kwargs={
        "temperature": 0.1,
    }
)

system_prompt = "Você é um assistente prestativo e está respondendo perguntas gerais"
user_prompt = "{input}"

token_s = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>system<|end_header_id|>"
token_e = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

prompt = ChatPromptTemplate.from_messages([
    ("system", token_s + system_prompt),
    ("user", user_prompt + token_e)
])

chain = prompt | llm

input = "Explique para mim brevemente o conceito de redes neurais, de forma clara e objetiva. Escreva em no máximo 1 parágrafo."


resp = chain.invoke({"input": input })
print(resp.content)