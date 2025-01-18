import os
import sys
import fitz
from glob import glob
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.schema import Document
import chardet
import urllib3
import pyocr
import pyocr.builders
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import ssl
import certifi
ssl_context = ssl.create_default_context()
ssl_context.load_verify_locations(certifi.where())

load_dotenv()

# Path設定
TESSERACT_PATH = 'C:/Program Files/Tesseract-OCR'  # インストールしたTesseract-OCRのpath
TESSDATA_PATH = 'C:/Program Files/Tesseract-OCR/tessdata'  # tessdataのpath
os.environ["PATH"] += os.pathsep + TESSERACT_PATH
os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH

# Initialize OCR tool
tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
ocr_tool = tools[0]

def detect_file_encoding(file_path):
    """
    Detect the encoding of a file.
    """
    with open(file_path, "rb") as f:
        raw_data = f.read(1024)  # Read the first 1KB to detect encoding
    result = chardet.detect(raw_data)
    return result["encoding"]

def PyMuPDFLoaderWithOCR(file_path):
    documents = []
    with fitz.open(file_path) as pdf:
        for page_number in range(len(pdf)):
            page = pdf[page_number]
            text = page.get_text("text")  # Extract text from the page

            if not text.strip():  # If text is empty, use OCR
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = ocr_tool.image_to_string(img, builder=pyocr.builders.TextBuilder())

            if text.strip():  # Add non-empty text as a Document
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "page_number": page_number + 1,
                        "source": file_path
                    }
                ))

    return documents

def initialize_vectorstore():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX"]
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    # Create index if it does not exist
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=os.environ.get("PINECONE_REGION", "us-west-1")
            )
        )
    return LangchainPinecone.from_existing_index(index_name, embeddings)

def get_loader(file_path):
    if file_path.endswith(".pdf"):
        return PyMuPDFLoaderWithOCR(file_path)
    elif file_path.endswith(".csv"):
        return CSVLoader(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return UnstructuredExcelLoader(file_path)
    else:
        # Detect file encoding and use UnstructuredLoader
        encoding = detect_file_encoding(file_path)
        return UnstructuredLoader(file_path, encoding=encoding)

if __name__ == "__main__":
    try:
        folder_path = "YOUR-DATA-PATH"
        file_paths = glob(os.path.join(folder_path, '*'), recursive=True)

        if not file_paths:
            print("No files found in the specified folder.")
            sys.exit()

        vectorstore = initialize_vectorstore()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        for file_path in file_paths:
            try:
                print(f"Processing file: {file_path}")
                loader = get_loader(file_path)

                # PDFや直接リストを返す場合の処理
                if isinstance(loader, list):  # リストの場合は直接使用
                    raw_docs = loader
                else:  # その他のローダーはload()を使用
                    raw_docs = loader.load()

                docs = text_splitter.split_documents(raw_docs)
                vectorstore.add_documents(docs)
                print(f"Successfully processed: {file_path}")
                print(len(docs))
            except Exception as file_error:
                print(f"Error processing {file_path}: {file_error}")
                print("Error! tmp stopp")
                break

        print("All files processed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
