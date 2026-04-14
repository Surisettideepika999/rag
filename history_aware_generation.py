from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

chat_history=[]
model=ChatOpenAI(model="gpt-4o", temperature=0.7)

db=Chroma(
    persist_directory="db/chroma",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_metadata={"hnsw:space": "cosine"}
)

retriver = db.as_retriever(search_kwargs={"k": 3})
def ask_quetion(query):

    if(chat_history):
        messages=[
            SystemMessage(content="You are a helpful assistant. Based on the attached history formulate given query to searchable question."),
            HumanMessage(content=f"History: {chr(10).join(chat_history)}\n\nQuestion: {query}\n\nSearchable question:")
        ]
        search_query=model.invoke(messages).content
        print("Formulating search query based on history...")
        print(f"new query: {search_query}")
    else:
        search_query=query
    resp_docs=retriver.invoke(search_query)

    combined_input = f"Based on the below retrieved documents, answer the question: {query}. docs: {chr(10).join([doc.page_content for doc in resp_docs])}\nDon't make up any information, if you don't find the answer in docs say you don't know"
    messages=[
        SystemMessage(content="You are a helpful assistant that answers questions based on retrieved documents. Just give the answers directly without any explanation."),
        HumanMessage(content=combined_input)    
    ]
    response = model.invoke(messages)
    chat_history.append(f"Question: {query}\nAnswer: {response.content}")
    return response.content

if __name__=="__main__":
    while True:
        query=input("Enter your question: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break
        answer=ask_quetion(query)
        print(f"Answer: {answer}\n")