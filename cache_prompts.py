import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("A chave GOOGLE_API_KEY não foi encontrada. Verifique o arquivo .env.")

model = GoogleGenerativeAI(
    model="gemini-1.5-pro",
    max_tokens=200
)

#set_llm_cache(InMemoryCache())

set_llm_cache(SQLiteCache(
    database_path='genai_cache.db'
))

prompt = 'Me diga quem foi nicolas tesla'

response1 = model.invoke(prompt)
print(f'Chamada 1: {response1}')

response2 = model.invoke(prompt)
print(f'Chamada 2: {response2}')
