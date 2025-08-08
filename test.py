import requests
import psycopg2
import re
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





def hierarchical_chunking(text: str):
    sections = text.split("\n\n")  # Split by double newline for sections
    hierarchy = []

    for section_id, section in enumerate(sections):
        paragraphs = section.strip().split("\n")
        for para_id, para in enumerate(paragraphs):
            sentences = para.strip().split(". ")
            for sent_id, sentence in enumerate(sentences):
                hierarchy.append({
                    "section_id": section_id,
                    "paragraph_id": para_id,
                    "sentence_id": sent_id,
                    "text": sentence.strip()
                })

    return hierarchy


# Step 1: Load the PDF using PyMuPDFLoader
loader = PyMuPDFLoader("testpdf.pdf")
documents = loader.load()


# Extract raw text from PDF
raw_text = "\n\n".join([doc.page_content for doc in documents])

# Apply hierarchical chunking
chunked_data = hierarchical_chunking(raw_text)
texts = [chunk["text"] for chunk in chunked_data]


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

retriever = vectorstore.as_retriever()

combined_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use both the chat history and the document context to answer the question./no_think. do not answer any question outside the document context"),
    ("human", "Chat History:\n{chat_history}\n\nContext:\n{context}\n\nQuestion:\n{input}")
])

document_chain = create_stuff_documents_chain(llm=llm, prompt=combined_prompt)

qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain
)
conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="Chat_History",
        user="postgres",
        password="Mh@2003"
    )

# Step 6: Save chat to PostgreSQL
def save_chat_to_postgres(user_message, bot_response):
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="Chat_History",
        user="postgres",
        password="Mh@2003"
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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(768),
            section_id INT,
            paragraph_id INT,
            sentence_id INT
        )
''' )


    
    for chunk, vector in zip(chunked_data, vectors):
        cursor.execute(
            "INSERT INTO documents (content, embedding, section_id, paragraph_id, sentence_id) VALUES (%s, %s, %s, %s, %s)",
                (chunk["text"], vector, chunk["section_id"], chunk["paragraph_id"], chunk["sentence_id"])
        ) 
        
    conn.commit()
    conn.close()      
    

def search_similar_chunks_pg(question_vector, top_k=5):
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="Chat_History",
        user="postgres",
        password="Mh@2003"
    )
    cursor = conn.cursor()

    vector_str = ','.join(map(str, question_vector))
    cursor.execute(query = f"""
        SELECT content, section_id, paragraph_id, sentence_id
        FROM documents
        ORDER BY embedding <-> CAST (ARRAY[{vector_str}] AS vector)
        LIMIT {top_k};
    """)


    results = cursor.fetchall()
    conn.close()
    return results

    

     
    

# Step 7: Ask a question and save the result
def clean_response(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def get_bot_response(user_input):
    # Generate embedding for the question
    question_vector = embedding_function.embed_query(user_input)

    # Retrieve similar chunks from PostgreSQL
    similar_chunks = search_similar_chunks_pg(question_vector)

    if not similar_chunks:
        return "Sorry, I can only answer questions related to the uploaded PDF document."

    # Format context from retrieved chunks
    context = "\n".join([chunk[0] for chunk in similar_chunks])  # chunk[0] is content

    '''# Run the LLM with context
    prompt = f"Chat History:\n{memory.load_memory_variables({})['chat_history']}\n\nContext:\n{context}\n\nQuestion:\n{user_input}"
    response = llm.invoke(prompt)
    '''
    # Load memory
    chat_history = memory.load_memory_variables({})["chat_history"]
    

    # Run the chain
    result = qa_chain.invoke({
        "input": user_input,
        "chat_history": chat_history,
        "context": context
        
    })
    
    answer = result["answer"]

    # Save to memory and database
    memory.save_context({"input": user_input}, {"answer": answer})
    save_chat_to_postgres(user_input, answer)

    return (answer)
