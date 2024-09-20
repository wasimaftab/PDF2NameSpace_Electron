from fastapi import FastAPI, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import sys
import os
import pinecone
import pymupdf  # PyMuPDF for PDF processing
import asyncio
import json

app = FastAPI()

##------- User defined function block -------##
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

def generate_embedding(text):
    # TODO: Replace with actual embedding logic
    # For example, using OpenAI embeddings
    # embedding = openai.Embedding.create(input=text)
    # return embedding['data'][0]['embedding']
    return [0.0] * 512  # Dummy embedding for placeholder
#########################################################

def process_pdfs(folder_path):
    if not os.path.exists(folder_path):
        print("Error: Folder path does not exist.")
        sys.exit(1)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                print(f'Processing {pdf_path}')
                sys.stdout.flush()

                try:
                    # Extract text from PDF
                    doc = pymupdf.open(pdf_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()

                    # Split the text recursively
                    texts = text_splitter.create_documents([text])

                    # Generate embedding
                    embedding = generate_embedding(text)

                    # Upsert to Pinecone
                    index.upsert([(file, embedding)])

                    print(f'Successfully processed {pdf_path}')
                    sys.stdout.flush()

                except Exception as e:
                    print(f'Error processing {pdf_path}: {e}')
                    sys.stdout.flush()
##------- User defined function block -------##

## Health check endpoint
@app.get("/health")
def read_health():
    return {"status": "OK"}

# Define a Pydantic model for the request body
class ProcessPDFsRequest(BaseModel):
    folder_path: str

# Define the /process-pdfs/ endpoint
@app.post("/process-pdfs/")
async def process_pdfs_endpoint(request: ProcessPDFsRequest):
    folder_path = request.folder_path
    process_pdfs(folder_path)
    return {"message": "Processing complete"}
    
# ... rest of your existing code ...
if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        process_pdfs(folder)
    else:
        print('No folder path provided.')
        sys.exit(1)



# # app.py
# from fastapi import FastAPI, HTTPException
# import sys
# import os
# import pinecone
# import pymupdf  # PyMuPDF for PDF processing
# import asyncio
# import json

# app = FastAPI()

# # check app status
# @app.get("/health")
# def read_health():
#     return {"status": "OK"}

# # Initialize Pinecone
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# # PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
# PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX')

# # if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT or not PINECONE_INDEX_NAME:
# if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
#     print("Error: Pinecone configuration not set.")
#     sys.exit(1)

# # pinecone.init(api_key=PINECONE_API_KEY)
# # index = pinecone.Index(PINECONE_INDEX_NAME)

# def generate_embedding(text):
#     # TODO: Replace with actual embedding logic
#     # For example, using OpenAI embeddings
#     # embedding = openai.Embedding.create(input=text)
#     # return embedding['data'][0]['embedding']
#     return [0.0] * 512  # Dummy embedding for placeholder

# def process_pdfs(folder_path):
#     if not os.path.exists(folder_path):
#         print("Error: Folder path does not exist.")
#         sys.exit(1)

#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith('.pdf'):
#                 pdf_path = os.path.join(root, file)
#                 print(f'Processing {pdf_path}')
#                 sys.stdout.flush()

#                 # try:
#                 #     # Extract text from PDF
#                 #     doc = pymupdf.open(pdf_path)
#                 #     text = ""
#                 #     for page in doc:
#                 #         text += page.get_text()

#                 #     # Generate embedding
#                 #     embedding = generate_embedding(text)

#                 #     # Upsert to Pinecone
#                 #     index.upsert([(file, embedding)])

#                 #     print(f'Successfully processed {pdf_path}')
#                 #     sys.stdout.flush()

#                 # except Exception as e:
#                 #     print(f'Error processing {pdf_path}: {e}')
#                 #     sys.stdout.flush()

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         folder = sys.argv[1]
#         process_pdfs(folder)
#     else:
#         print('No folder path provided.')
#         sys.exit(1)
