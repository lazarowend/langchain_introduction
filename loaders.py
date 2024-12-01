import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
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

loader = TextLoader('base_conhecimento.txt')
documents = loader.load()

prompt_base_conhecimento = PromptTemplate(
    input_variables=['contexto', 'pergunta'],
    template='''
    Use o seguinte contexto para respoder a pergunta.
    Responda apenas com base nas informaçoes fornecidas
    não ultilize informacoes externas ao contexto:
    Contexto: {contexto}
    Pergunta: {pergunta}
    '''
)

chain = prompt_base_conhecimento | model | StrOutputParser()

response = chain.invoke(
    {
        'contexto': documents,
        'pergunta': 'quantos nucleos e threads tem o xeon 2673v3?'
    }
)

print(response)