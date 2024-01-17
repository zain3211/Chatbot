import os
from pathlib import Path
from dotenv import load_dotenv
import zipfile
from docx2txt import process
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
import openai
import streamlit as st
from transformers import AutoModel, AutoTokenizer
from langchain.retrievers import MultiQueryRetriever  # Import MultiQueryRetriever

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY or not QDRANT_API_KEY:
    raise ValueError("OpenAI or Qdrant API key not found. Please set the environment variables.")

openai.api_key = OPENAI_API_KEY

# Set up your Qdrant connection
qdrant_connection = QdrantClient(
    url="https://55b2c532-f1fc-4d1f-a0a0-913446b6bbdf.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key=QDRANT_API_KEY,
)

# Loaders for PDF and ZIP files
def load_text_from_pdf_pypdf(file_path):
    from PyPDF2 import PdfReader

    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def load_text_from_docx(file_path):
    return process(file_path)

def load_text_from_zip(zip_path):
    text_content = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for filename in zip_ref.namelist():
            if filename.endswith(".pdf"):
                text = load_text_from_pdf_pypdf(zip_ref.extract(filename))
            elif filename.endswith(".docx"):
                text = load_text_from_docx(zip_ref.extract(filename))
            else:
                print(f"Unsupported file format: {filename}")
                continue
            text_content.append(text)
    return text_content

# Create or get the Qdrant collection
def create_or_get_qdrant_collection(connection, collection_name):
    try:
        info = connection.get_collection(collection_name)
        print("Collection already exists.")
    except Exception as e:
        if "not found" in str(e).lower():
            connection.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )
            print("Create collection response:", connection)
        else:
            raise e

# Text chunking functions using different splitters
def get_text_chunks_character(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_text_chunks_recursive(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", " "],  # Split by newline and space
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Generate embeddings using Hugging Face models
def generate_embeddings(chunks, model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(chunks, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over tokens

    return embeddings.tolist()

# Insert embeddings into Qdrant
def insert_embeddings_into_qdrant(collection, embeddings, text_chunks):
    for text, embedding in zip(text_chunks, embeddings):
        doc = {"text": text, "vector": embedding}
        collection.upsert(points=[PointStruct(**doc)])

# Main processing logic
def process_file(file_path, connection, collection_name):
    try:
        # Save the uploaded file temporarily
        temp_dir = Path("temp_files")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / file_path.name
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_path.read())

        if temp_file_path.name.endswith(".pdf"):
            text = load_text_from_pdf_pypdf(temp_file_path)
        elif temp_file_path.name.endswith(".docx"):
            text = load_text_from_docx(temp_file_path)
        elif temp_file_path.name.endswith(".zip"):
            text_content = load_text_from_zip(temp_file_path)
            text = "\n".join(text_content)
        else:
            st.error("Unsupported file format.")
            return
        st.success("Processing completed successfully!")

        query = st.text_input("Enter your query:")
        if query:
            # Use the correct endpoint for chat models
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query},
                ],
            )
            answer = response['choices'][0]['message']['content'].strip()
            st.write("Answer:", answer)

        chunks_character = get_text_chunks_character(text)
        chunks_recursive = get_text_chunks_recursive(text)

        for chunks, splitter_name in [(chunks_character, "character"), (chunks_recursive, "recursive")]:
            # Use "BAAI/bge-small-en-v1.5" model for one splitter
            embeddings_1 = generate_embeddings(chunks, "BAAI/bge-small-en-v1.5")
            insert_embeddings_into_qdrant(qdrant_connection, embeddings_1, chunks)

            # Use "'intfloat/e5-large-v2'" model for the other splitter
            embeddings_2 = generate_embeddings(chunks, "intfloat/e5-large-v2")
            insert_embeddings_into_qdrant(qdrant_connection, embeddings_2, chunks)

        st.success("Processing completed successfully!")

        # Use MultiQueryRetriever
        retriever = MultiQueryRetriever(
            retrievers=[
                {"embeddings": embeddings_1, "model_name": "BAAI/bge-small-en-v1.5"},
                {"embeddings": embeddings_2, "model_name": "intfloat/e5-large-v2"},
            ],
        )

        query_result = retriever.retrieve(query)
        st.write("Results from MultiQueryRetriever:")
        for result in query_result:
            st.write(result)

    except Exception as e:
        st.error(f"Error processing file: {e}")

    finally:
        # Clean up: Remove the temporary file
        temp_file_path.unlink(missing_ok=True)

# Streamlit application
def main():
    st.title("Transforming Text into Knowledge with LangChain's Power")

    file_path = st.file_uploader("Upload a PDF, DOCX, or ZIP file")

    if file_path:
        create_or_get_qdrant_collection(qdrant_connection, "zain")
        process_file(file_path, qdrant_connection, "zain")

    query = st.text_input("Enter your query:")

    if query:
        # Generate the query embedding
        query_embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5").encode([query])

        # Perform retrieval using the Qdrant vector store
        results = qdrant_connection.search(
            collection_name="zain",
            vector=query_embedding,
            limit=10,
        )

        # Display the results
        st.write("Results:")
        for result in results:
            st.write(result)

if __name__ == "__main__":
    main()
