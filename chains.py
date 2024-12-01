import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("A chave GOOGLE_API_KEY não foi encontrada. Verifique o arquivo .env.")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1,
    max_tokens=200,
    timeout=None,
    max_retries=2,
)

prompt_template = PromptTemplate.from_template(
    'me fale sobre o carro {carro}'
)

runnable_sequence = prompt_template | model | StrOutputParser()

response = runnable_sequence.invoke(
    {'carro': 'siena 2008'}
)

print(response)