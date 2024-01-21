import chromadb
from openai import OpenAI
import pandas as pd
import ast
import time
import os
from dotenv import load_dotenv
import requests


csv_file_path = 'embeddings.csv'
df = pd.read_csv(csv_file_path)

embeddings = []
documents = []
for index, row in df.iterrows():
    embedding_str = row['Embedding']
    embedding_list = ast.literal_eval(embedding_str)
    embeddings.append(embedding_list)
    documents.append(row['Document'])



# Loading embeddings in ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("test")
collection.add(
    embeddings=embeddings,
    documents=documents,
    ids = [str(i) for i in range(len(embeddings))]
)


load_dotenv()
openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
model = "gpt-3.5-turbo-1106"



def get_embedding(text, client,model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   time.sleep(20)
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# To search similar content related to query
def semantic_search(query):
  query_result = collection.query(
        query_embeddings= get_embedding(query,openai_client),
        n_results=5,
    )
  return (query_result['documents'][0])

query = "When was sam altman born?"
similar_content = ' '.join(semantic_search(query))


# Rephrase content to be more meaningful
def content_rephrase(client,text):
  system_prompt ="""You are given a text.
  You have to rephrase the text to make it meaningful.
  No need to uncessarily expand the text.
    """
  try:
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Correct this {text}"}
      ]
    )
  except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
  return (completion.choices[0].message.content)

similar_content = content_rephrase(openai_client,similar_content)

# Final output
def query_output(client,query):
  system_prompt ="""You are given a user query along with the data which was retrived from documents which user uploaded.
  Answer user query strictly based on this data and no other external knowledge.
  If you don't have knowledge regarding query say "Insufficient Data"
    """
  model = "gpt-3.5-turbo-1106"
  try:
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
      ]
    )
  except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
  return (completion.choices[0].message.content)


final_user_query = "Query:" + query + "Retrived data:" + similar_content
print(query_output(openai_client,final_user_query))
