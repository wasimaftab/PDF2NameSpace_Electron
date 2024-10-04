import os
os.system('clear')

from fastapi import FastAPI, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import sys
from pinecone import Pinecone
import pymupdf  # PyMuPDF for PDF processing
import asyncio
import json
import glob
import requests
import re
import xml.etree.ElementTree as ET
from grobid_client_python.grobid_client.grobid_client import GrobidClient
from tqdm import tqdm
import numpy as np
import torch
from openai import OpenAI
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import logging


# import python.PMC_downloader_Utils.py as pmcd
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
## Define the namespace
namespaces = {'ns': 'http://www.tei-c.org/ns/1.0'}

## Setup logger
# log_dir = "/home/wasim/Desktop/QA_Bot_Web_App/App/Logs"
log_dir = os.getcwd() + "/Logs"
log_file = "PDF2NS.log"   
logger_name = 'PDF2NS_logger'

## Start the fastapi app    
app = FastAPI()

##------- User defined function block -------##

def create_logger(log_dir, log_file, logger_name):
    # Name your logger 
    logger = logging.getLogger(logger_name) 
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create log dir and file if does not exist
    log_path = os.path.join(log_dir, log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a file handler that logs debug and higher level messages
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    # Specify the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.INFO)  
        
    return logger  

###############################################################################
## Create a global variable to log across different places in this file
logger = create_logger(log_dir, log_file, logger_name)
###############################################################################    

def get_pinecone_index():
    ## New approach After pinecone-client version ≥ 3.0.0
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])  
    index = pc.Index('namespaces-in-paper')
    return index

def beautify_title(title):
    # Remove XML tags
    clean_title = re.sub(r'<.*?>', '', title)  # Removes all XML tags
    return clean_title.strip()  # Strip leading/trailing whitespace

def extract_citation_info(root):
    # Extract citation information
    publisher_element = root.find('.//ns:monogr/ns:imprint/ns:publisher', namespaces)
    date_element = root.find('.//ns:monogr/ns:imprint/ns:date[@type="published"]', namespaces)
    # doi_element = root.find('.//ns:idno[@type="DOI"]', namespaces)

    # Extract first author's name
    first_author = root.find('.//ns:analytic/ns:author/ns:persName', namespaces)
    if first_author is not None:
        surname = first_author.find('ns:surname', namespaces)
        given_name = first_author.find('ns:forename', namespaces)
        author_name = f"{given_name.text} {surname.text}" if given_name is not None and surname is not None else 'N/A'
        authors = author_name + ' et al.'
    else:
        authors = 'N/A'

    publisher = publisher_element.text if publisher_element is not None else 'N/A'
    date = date_element.text if date_element is not None else 'N/A'
    # doi = doi_element.text if doi_element is not None else 'N/A'
    
    return authors, publisher, date

def iter_paragraphs(paragraph):
    """Iterates over a paragraph and its children to extract all text."""
    if paragraph.text:
        yield paragraph.text
    for child in paragraph:
        yield from iter_paragraphs(child)
        if child.tail:
            yield child.tail

def remove_unwanted_spaces(text):
    # The regex to match
    pattern = re.compile(r"(?<=\s|\)|\])(\b[\w-]+\s+et al\.\s\(\d{4}\)|\(\s*[^\d\(]*\d{4}(?:;[^\d\(]*\d{4})*\s*\)|\((?:\d+(?:-\d+)?)(?:,\s*(?:\d+(?:-\d+)?))*\)|\[\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*\])")

    # Replace the citations
    text_without_citations = re.sub(pattern, '', text)

    # Post-process text to remove multiple spaces
    text_without_citations = re.sub(' +', ' ', text_without_citations)

    # Trim spaces around punctuation
    text_without_citations = re.sub(r'\s+([,.])', r'\1', text_without_citations)
    
    return text_without_citations.strip()

def remove_newline_multiple_spaces(string):
    string = string.replace("\n", " ")
    string = re.sub(' +', ' ', string).strip()
    return string

def is_grobid_server_running(url="http://localhost:8070"):
    try:
        response = requests.get(f"{url}/api/isalive")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

## Updated for pinecone-client version ≥ 3.0.0
def get_num_vectors_in_namespace(namespace):
   
    # Connect to an existing index
    index = get_pinecone_index()
    
    # Get number of vectors in the namespace
    index_stats = index.describe_index_stats()
    num_vectors = index_stats.get('namespaces')[namespace].vector_count
    
    # logger.info(f"Number of vectors in namespace {namespace}: {num_vectors}")   
    return num_vectors

## Compatible with pinecone-client version ≥ 3.0.0
def push_vectors_into_pinecone(all_texts_citated, namespace, embedd_model):
    # namespace += '_' + embedd_model
    # pdb.set_trace()
    logger.info("Inside push_vectors_into_pinecone()")
    # print("Inside push_vectors_into_pinecone()")
    
    # initialize pinecone and connect to an index
    index = get_pinecone_index()
    
    ## Get the existing number of vectors in a namespace
    index_stats = index.describe_index_stats()
    if namespace in index_stats["namespaces"]:
        offset = get_num_vectors_in_namespace(namespace)
        logger.info(f"Number of vectors in namespace {namespace}: {offset}")
        print(f"Number of vectors in namespace {namespace}: {offset}")
    else:
        offset = 0
        logger.info(f"The namespace {namespace} does not yet exist, will be created, hence in the beginning number of vectors in namespace {namespace} is: {offset}")
        # print(f"The namespace {namespace} does not yet exist, will be created, hence number of vectors in namespace {namespace} will be: {offset}")
                          
    batch_size = 32  # process everything in batches of 32
    
    for i in tqdm(range(0, len(all_texts_citated), batch_size)):   
        # pdb.set_trace()                   
        # print(f"\ni = {i}\n")
        
        # set end position of batch
        i_end = min(i+batch_size, len(all_texts_citated))

        # get batch of lines and IDs
        records_batch = all_texts_citated[i: i+batch_size]

        # ids_batch = [str(n) for n in range(i, i_end)]
        ids_batch = [str(n + offset + 1) for n in range(i, i_end)]        

        texts_batch = [record.page_content  for record in records_batch]
        # citations_batch = [record.metadata['citation']  for record in records_batch]
        
        if bool(texts_batch):
            ## Create embeddings 
            # embeds = get_text_embeddings(texts_batch, embedd_model, openai_vec_len=1536)
            embeds = get_text_embeddings('chunk', texts_batch, embedd_model, openai_vec_len=1536)
            
            # prep metadata and upsert batch
            meta = [{'text': text} for text in texts_batch]
            cite = [{'citation': record.metadata['citation']} for record in records_batch]
            paper_id = [{'paper_id': record.metadata['paper_id']} for record in records_batch]
            chunk_id = [{'chunk_id': record.metadata['chunk_id']} for record in records_batch]
            
            
            
            if len(meta) == len(cite):
                for i, dict_item in enumerate(meta):
                    dict_item['citation'] = cite[i]['citation']
                    dict_item['paper_id'] = paper_id[i]['paper_id']
                    dict_item['chunk_id'] = chunk_id[i]['chunk_id']
            else:
                msg = "Lists have different lengths, cannot add citations to pinecone meta. Run terminated."
                logger.info(msg)
                return {"code" : "failure", "msg" : msg}
            
            to_upsert = zip(ids_batch, embeds, meta)
            # pdb.set_trace()
        else:
            ## Empty text in records_batch
            # import sys
            # sys.exit('Empty text in records_batch')
            msg = "Empty text in records_batch"
            return {"code" : "failure", "msg" : msg}

        ## upsert to Pinecone            
        # logger.info(f"Pushing {len(ids_batch)} vectors into Pinecone")     
        logger.info(f"Pushing {len(ids_batch)} vectors into Pinecone")
        try:
            index.upsert(vectors=list(to_upsert), namespace=namespace)
        except Exception as e:
            msg = f"Error pushing the vectors in to pinecone, actual exception is: {e}"
            
            
    # pdb.set_trace()
            
    # Close the index and return (NOT REQUIRED in pinecone-client version ≥ 3.0.0)
    # index.close()

#########################################################

## For WeiseEule paper revision
def return_embeddings(lines_batch, model, tokenizer, max_length, openai_vec_len):
    embeds = []
    for text in lines_batch:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
        # pdb.set_trace()
        outputs = outputs.last_hidden_state.mean(dim=1).numpy()            
        outputs_padded = np.pad(np.squeeze(outputs), (0, openai_vec_len - outputs.shape[1]), 'constant', constant_values=0)
        embeds.append(outputs_padded.tolist())
    return embeds 

    
def get_text_embeddings(text_type, lines_batch, embedd_model, openai_vec_len):
    if embedd_model == "openai":
        # res = openai.Embedding.create(
        #     input=lines_batch, engine=EMBEDDING_MODEL)
        embeds = []
        for text in lines_batch:
            res = syncClient.embeddings.create(input=text, model=EMBEDDING_MODEL)
            embeds.append(res.data[0].embedding)

    elif embedd_model == "biobert":
        biobert_model = "dmis-lab/biobert-base-cased-v1.2"
        tokenizer = BertTokenizer.from_pretrained(biobert_model)
        model = BertModel.from_pretrained(biobert_model)
        max_length = 512
        
        embeds = return_embeddings(lines_batch, model, tokenizer, max_length, openai_vec_len)

    elif embedd_model == "MedCPT":
        if text_type == 'chunk':
            max_length = 512
            logger.info(f"Inside get_text_embeddings(), text_type = {text_type} and max_length = {max_length}")
            model_name = "ncbi/MedCPT-Article-Encoder"
        else:
            max_length = 64
            logger.info(f"Inside get_text_embeddings(), text_type = {text_type} and max_length = {max_length}")
            model_name = "ncbi/MedCPT-Query-Encoder"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        embeds = return_embeddings(lines_batch, model, tokenizer, max_length, openai_vec_len)
      
    return embeds
###############################################################################

def process_pdfs(folder_path, namespace):
    embedd_model = "MedCPT"
    if not os.path.exists(folder_path):
        print("Error: Folder path does not exist.")
        sys.exit(1)

    print(f"Selected folder = {folder_path}")
    print(f"parent_directory = {parent_directory}")

    if is_grobid_server_running():
        print("GROBID server is up and running")
    else:
        print("GROBID server is not running")

    client = GrobidClient(config_path="./grobid_client_python/config.json")
    client.process("processFulltextDocument", 
                    input_path=folder_path, 
                    output="./pdf2xml",
                    n=10,  # may want to adjust this based on the cores available
                    force=True)

    xml_folder_path = "./pdf2xml"
    # xml_folder_path = "./pdf2xml_old"
    
    ## Get a list of xml files and extract pmids from file names
    xml_files = glob.glob(xml_folder_path + "/*.tei.xml")
    print(f"xml_files = {xml_files}")

    ## Create a list of headers upon hitting which code must stop extracting
    stop_headers = ['acknowledgement', 'references', 'funding', 'availability']
    full_text = []
    citation_text = []
    
    for idx, xml_file in enumerate(xml_files):
        # temp_xml_file = os.path.basename(xml_file)
        # tree = ET.parse(xml_folder_path + '/' + temp_xml_file)
        tree = ET.parse(xml_file)

        # Extract title
        root = tree.getroot()
        title_element = root.find('.//ns:title', namespaces)
        title = title_element.text if title_element is not None else 'N/A'

        # Beautify the title
        clean_title = beautify_title(title)
        
        # Extract citation information
        authors, publisher, date = extract_citation_info(root)

        # Format the citation
        formatted_citation = f"{authors}, '{clean_title}', {publisher}, {date}"
        citation_text.append(formatted_citation)

        ## Build a dictionary to map children to their parent
        parent_map = {c: p for p in tree.iter() for c in p}
        
        ## Find all paragraph elements
        paragraphs = tree.findall('.//ns:p', namespaces)
        
        ## Extract and print the text from each paragraph that's not inside an 'acknowledgement' div
        all_text = ""
        abstract_text = ""
        
        stop_flag = False

        for p in paragraphs:
            parent = parent_map.get(p)
            skip = False
            while parent is not None:
                if parent.tag == '{http://www.tei-c.org/ns/1.0}profileDesc':
                    for abstract in parent.findall('.//ns:abstract', namespaces):
                        if abstract is not None:
                            abstract_text += p.text + "\n"  # Add abstract text
                            skip = True  # Skip the abstract section
                            break  # Stop going up the tree
                elif parent.tag == '{http://www.tei-c.org/ns/1.0}div' and parent.attrib.get('type') in stop_headers:
                    stop_flag = True
                    break
                parent = parent_map.get(parent)
    
            if stop_flag:
                break
            elif skip:
                continue
            else:
                text = ''.join(iter_paragraphs(p))
                if len(text.split()) < 10:
                    continue
                all_text += text + "\n"
        all_text = abstract_text + all_text
        all_text = all_text.replace("\n", " ").lower()        
        all_text = remove_newline_multiple_spaces(remove_unwanted_spaces(all_text))
        # if all_text:
        #     full_text.append(all_text)
        # print(f"############# full text from {xml_file} #############\n")
        citation = f"{formatted_citation} [Source: {folder_path + '/' + os.path.basename(xml_file).replace('.grobid.tei.xml', '.pdf')}]"
        # print(f"{full_text}\n")
        # Split the text recursively
        texts = text_splitter.create_documents([all_text])
        for chunk_idx, chunk in enumerate(texts):
                    chunk.metadata['chunk_id'] = chunk_idx
                    chunk.metadata['citation'] = citation
                    chunk.metadata['paper_id'] = idx+1
        push_vectors_into_pinecone(texts, namespace, embedd_model)
        print(f"Just pushed {len(texts)} vectors into pinecone")


    # for root, dirs, files in os.walk(xml_folder_path):
    #     for file in files:
    #         if file.lower().endswith('.xml'):
    #             xml_path = os.path.join(root, file)
    #             print(f'Processing {xml_path}')

                # sys.stdout.flush()

                # try:
                #     # Extract text from PDF (update the code to use GROBID)
                #     doc = pymupdf.open(pdf_path)
                #     text = ""
                #     for page in doc:
                #         text += page.get_text()

                #     # Split the text recursively
                #     texts = text_splitter.create_documents([text])

                #     # Generate embedding
                #     embedding = generate_embedding(text)

                #     # Upsert to Pinecone
                #     # index.upsert([(file, embedding)])

                #     print(f'Successfully processed {pdf_path}')
                #     sys.stdout.flush()

                # except Exception as e:
                #     print(f'Error processing {pdf_path}: {e}')
                #     sys.stdout.flush()
##------- User defined function block -------##

## Health check endpoint
@app.get("/health")
def read_health():
    return {"status": "OK"}

# Define a Pydantic model for the request body
class ProcessPDFsRequest(BaseModel):
    folder_path: str
    namespace: str

# Define the /process-pdfs/ endpoint
@app.post("/process-pdfs/")
async def process_pdfs_endpoint(payload: ProcessPDFsRequest):
    folder_path = payload.folder_path
    namespace = payload.namespace
    process_pdfs(folder_path, namespace)
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
