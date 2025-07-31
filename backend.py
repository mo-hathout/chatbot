import requests
import psycopg2
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader





#with open ('testpdf.pdf','r',encoding = 'utf-8') as file:
           #text=file.read()

#text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100,chunk_overlap = 0,separator='',#strip_whitespace=False)
#documents = text_splitter.create_documents([text])
#print(documents)          

# Step 1: Load the PDF file
#loader = PyMuPDFLoader("testpdf.pdf")
#documents = loader.load()
#texts = [doc.page_content for doc in documents]    
#texts = [doc.page_content for doc in text_splitter.create_documents([text])]



# Step 1: Load the PDF using PyMuPDFLoader
loader = PyMuPDFLoader("testpdf.pdf")
documents = loader.load()

# Step 2: Extract text and split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    strip_whitespace=True
)

# Apply the splitter to the loaded documents
split_documents = text_splitter.split_documents(documents)

# Extract the text chunks
texts = [doc.page_content for doc in split_documents]

print(texts)




# Step 2: Send content to LM Studio's embedding endpoint
embedding_url = "http://localhost:1234/v1/embeddings"
model_name = "text-embedding-nomic-embed-text-v1.5@q4_k_m"

embedding_payload = {
    "input": texts,
    "model": model_name
}

response = requests.post(
    embedding_url,
    headers={"Content-Type": "application/json"},
    json=embedding_payload
)

if response.status_code != 200:
    print("Embedding request failed:", response.status_code)
    print("Response:", response.text)
    exit()

response_json = response.json()
if "data" not in response_json:
    print("Unexpected response format:", response_json)
    exit()

embedding_data = response_json["data"]
vectors = [item["embedding"] for item in embedding_data]

print("Number of embeddings:", len(vectors))
print("Embedding dimension:", len(vectors[0]))
print("Sample embedding vector (first 10 values):", vectors[0][:10])



# Step 3: Create FAISS vector store using custom embedding class
class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return vectors

    def embed_query(self, text):
        query_response = requests.post(
            embedding_url,
            headers={"Content-Type": "application/json"},
            json={"input": [text], "model": model_name}
        )

        if query_response.status_code != 200:
            print("Query embedding request failed:", query_response.status_code)
            print("Response:", query_response.text)
            return []

        response_json = query_response.json()
        if "data" not in response_json:
            print("Unexpected query embedding response format:", response_json)
            return []

        return response_json["data"][0]["embedding"]

embedding_function = CustomEmbeddings()
vectorstore = FAISS.from_texts(texts, embedding=embedding_function)

# Step 4: Connect to LM Studio's chat model
llm = ChatOpenAI(
    model_name="qwen3-4b",
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed",
    temperature=0.7
)

# Step 5: Create history-aware retrieval chain
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

question_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

retriever = create_history_aware_retriever(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    prompt=question_prompt
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context below."),
    ("human", "Context:\n{context}\n\nQuestion:\n{input}")
])

document_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain
)

# Step 6: Save chat to PostgreSQL
def save_chat_to_postgres(user_message, bot_response):
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="Chat_History",
        user="postgres",
        password="CorpCap@2025"
    )
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute(
        'INSERT INTO chat_history (question, answer) VALUES (%s, %s)',
        (f"User: {user_message}", f"Bot: {bot_response}")
    )
    
    for chunk, vector in zip(texts, vectors):
        cursor.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (chunk, vector)
        )

    conn.commit()
    conn.close()

# Step 7: Ask a question and save the result
def get_bot_response(user_input):
    chat_history = memory.load_memory_variables({})["chat_history"]
    result = qa_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    answer = result["answer"]
    memory.save_context({"input": user_input}, {"answer": answer})
    save_chat_to_postgres(user_input, answer)
    return answer

