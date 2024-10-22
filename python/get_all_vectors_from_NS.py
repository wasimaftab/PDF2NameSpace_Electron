######################## Testing: extract all vectors from pinecone #########################
import numpy as np
# import QA_OpenAI_api_Utils as  qa
# import PMC_downloader_Utils as pmcd
import copy
import time
import pickle
import json
import sys
import os
from pinecone import Pinecone

## Select LLM and set temp
# llm = "gpt-3.5-turbo"
llm = "gpt-4"
temp = 0

## Select word embedding model
# embedd_model='biobert'
# embedd_model='openai'
embedd_model='MedCPT'
# namespace = 'some_namespace'
namespace = 'vv_namespace'

# namespace = 'swr1_nua4_meiosis_1_key_paper_art_en_MedCPT'


# query = "Find all results that connect msl-2 complex with dosage compensation?"
# pdb.set_trace()
def remove_values_key_pinecone_res(res):
    data_without_values = []

    for d in res:
        # Deep copy the dictionary to prevent modifying the original object
        dict_representation = copy.deepcopy(d.__dict__)

        # Access the nested _data_store dictionary
        data_store = dict_representation.get('_data_store', {})

        # Remove 'values' key if it exists inside _data_store
        if 'values' in data_store:
            del data_store['values']

        # Create the dictionary with the desired order
        ordered_data_store = {
            'id': data_store.get('id'),
            'metadata': data_store.get('metadata'),
            'score': data_store.get('score')
        }

        # Append to the new list
        data_without_values.append(ordered_data_store)
    return data_without_values

def get_pinecone_index():
    ## New approach After pinecone-client version ≥ 3.0.0
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])  
    index = pc.Index('test-index')
    return index

def get_matches_from_query(index, input_vector, namespace):
    print("\nsearching pinecone...")
    results = index.query(vector=input_vector, top_k=1000, include_metadata=True, namespace=namespace, include_values=False)
    # results = index.query(vector=input_vector, top_k=5, include_metadata=True, namespace=namespace, include_values=False)

    ids = set()
    print(type(results))
    for result in results['matches']:
        ids.add(result['id'])
    return {"matches": results['matches'], "ids": ids}


def get_all_matches_from_index(index, num_dimensions, namespace=""):
    #print(index.describe_index_stats())
    # num_vectors = index.describe_index_stats(
    # )["total_vector_count"]#[namespace]['vector_count']
    
    # Get number of vectors in the namespace
    index_stats = index.describe_index_stats()
    num_vectors = index_stats.get('namespaces')[namespace].vector_count    
    all_matches = []
    all_ids = set()
    prev_ids = set()
    ids = set()
    prev_matches = set()
    loop = 0
    while len(all_ids) < num_vectors:
        print("Length of ids list is shorter than the number of total vectors...")
        input_vector = np.random.rand(num_dimensions).tolist()
        print("creating random vector...")
        res = get_matches_from_query(index, input_vector, namespace)
        all_matches.append(res['matches'])
        ids = res['ids']
        # pdb.set_trace()

        # print("getting ids from a vector query...")
        prev_ids = copy.deepcopy(all_ids)
        all_ids.update(ids)
        # print("updating ids set...")
        if loop:
            # pdb.set_trace()
            delta_fetch = all_ids.difference(prev_ids)
            print(f"{len(delta_fetch)} new ids are fetched in loop {loop+1}")
        loop += 1
        print(f"Collected {len(all_ids)} ids out of {num_vectors}.")
        # pdb.set_trace()
        time.sleep(10)
        
    ## convert list of lists to list
    records = []
    for alist in all_matches:
        for l in alist:
            records.append(l)
    return {"all_matches": records, "all_ids": all_ids}

index = get_pinecone_index()

matches_ids_dict = get_all_matches_from_index(index, num_dimensions=1536, namespace=namespace)

## Now remove redundant rocords
seen_ids = set()
unique_matches = []

for d in matches_ids_dict['all_matches']:
    id_val = d['id']
    if id_val not in seen_ids:
        seen_ids.add(id_val)
        unique_matches.append(d)

## There are empty values field, first remove that from each dict
unique_matches_empty_removed = remove_values_key_pinecone_res(unique_matches)

## Check indeed it has all the records
unique_ids = []
for match in unique_matches_empty_removed:
    unique_ids.append(match['id'])
    
if sorted(matches_ids_dict['all_ids']) == sorted(unique_ids):
    print('unique_matches_empty_removed contains all the unique records')
else:
    sys.exit('unique_matches_empty_removed does NOT contain all the unique records')

# ## Now save the unique_matches
# with open(namespace + '_chunks.pkl', 'wb') as f:
#     pickle.dump(unique_matches_empty_removed, f)
    
# ## To load the unique_matches
# with open(namespace + '_chunks.pkl', 'rb') as f:
#     chunks_in_namespace = pickle.load(f)    
# https://docs.pinecone.io/docs/manage-data
# index.fetch(list(all_ids))

##############################################################################