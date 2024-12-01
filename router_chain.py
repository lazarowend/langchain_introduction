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

classification_chain = (
    PromptTemplate.from_template(
        '''
        Classifique a pergunta do usuario em um dos seguintes setores:
        - Financeiro
        - Suporte Técnico
        - Outras Informações
        
        Pergunta: {pergunta}
        '''
    )
    | model
    | StrOutputParser()
)

financial_chain = (
    PromptTemplate.from_template(
        '''
        Você é um especialista financeiro.
        Sempre inicie as conversas com "Bem-vindo(a) ao setor Financeiro"
        Responda a pergunta do usuario:        
        Pergunta: {pergunta}
        '''
    )
    | model
    | StrOutputParser()
)

tech_suport_chain = (
    PromptTemplate.from_template(
        '''
        Você é um especialista no suporte técnico.
        Sempre inicie as conversas com "Bem-vindo(a) ao setor de Suporte Técnico"
        Responda a pergunta do usuario:        
        Pergunta: {pergunta}
        '''
    )
    | model
    | StrOutputParser() 
)

outher_info_chain = (
    PromptTemplate.from_template(
        '''
        Você é um especialista em informaçoes gerais.
        Sempre inicie as conversas com "Bem-vindo(a) ao setor de Informações Gerais"
        Responda a pergunta do usuario:        
        Pergunta: {pergunta}
        '''
    )
    | model
    | StrOutputParser() 
)

def route(classification):
    classification = classification.lower()
    if 'financeiro' in classification:
        return financial_chain
    elif 'técnico' in classification:
        return tech_suport_chain
    else:
        return outher_info_chain

pergunta = input('Qual a sua pergunta?: ')

classification = classification_chain.invoke(
    {'pergunta': pergunta}
)

response_chain = route(classification=classification)

response = response_chain.invoke(
    {'pergunta': pergunta} 
)

print(response)
