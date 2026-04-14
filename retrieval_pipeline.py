from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

persistant_path = "db/chroma"
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

db =Chroma(
    persist_directory=persistant_path,
    embedding_function=embedding,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "What are 2 mostly visited websites in the world?"
retriever = db.as_retriever(search_kwargs={"k": 3})

resp_docs=retriever.invoke(query)

# for i, doc in enumerate(resp):
#     print(f"result - {i+1}")
#     print(f"metadata: {doc.page_content}...")

model= ChatOpenAI(model="gpt-4o", temperature=0.7)

combined_input = f"Based on the below retrieved documents, answer the question: {query}. docs: {chr(10).join([doc.page_content for doc in resp_docs])}\nDon't make up any information, if you don't find the answer in docs say you don't know"

messages=[
    SystemMessage(content="You are a helpful assistant that answers questions based on retrieved documents. Just give the answers directly without any explanation."),
    HumanMessage(content=combined_input)
]

response = model.invoke(messages)
print(f"Answer: {response.content}")

