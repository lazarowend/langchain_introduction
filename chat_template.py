import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


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


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1,
    max_tokens=200,
    timeout=None,
    max_retries=2,
)

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content='Voce deve responder baseado em dados geograficos dae regios do brasil.'),
        HumanMessagePromptTemplate.from_template('por favor me fale sobre a regiao {regiao}'),
        AIMessage(content='claro, vou começar coletando informacoes sobre a regiao e analisando os dados disponiveis'),
        HumanMessage(content='certifique de incluir os dados demograficos'),
        AIMessage(content='entendido, aqui estao os dados: ')
    ]
)

prompt = chat_template.format_messages(
    regiao='Sudeste'
)

response = model.invoke(
    prompt
)

print(response.content)