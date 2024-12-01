import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("A chave GOOGLE_API_KEY não foi encontrada. Verifique o arquivo .env.")

# model = GoogleGenerativeAI(
#     model="gemini-1.5-pro",
# )

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1,
    max_tokens=500,
    timeout=None,
    max_retries=2,
)

messages = [
    {'role': 'system', 'content': 'Você é um assistente de uma oficina de mecanica de carros que fornece informaçôes técnicas sobre os veiculos e mecanicas'},
    {'role': 'user', 'content': 'especificacoes do motor do siena 2008, motor fire 1.0'}
]

response = model.invoke(messages)

print(response)
