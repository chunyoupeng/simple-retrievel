from operator import itemgetter
import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

os.environ['OPENAI_API_KEY'] = "sk-Ny6WUAgn9PQCOMqQ0d9a0174Ba9e45348862D2746aF44923"
os.environ['OPENAI_API_BASE'] = "https://aiapi.xing-yun.cn/v1"

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFDirectoryLoader("src/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(docs)
db = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = db.as_retriever()


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

while True:
    question = input("请输入你的问题：")
    out = chain.invoke(question)
    print(out)
