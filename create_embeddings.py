import pandas as pd
from openai import OpenAI
import os
import requests
from bs4 import BeautifulSoup
import time
from dotenv import load_dotenv





def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()


        soup = BeautifulSoup(response.text, 'html.parser')

        text = ' '.join([p.get_text() for p in soup.find_all('p')])

        return text

    except Exception as e:
        print(f"Error extracting text from {url}: {e}")
        return None


# Break long text into small chunks
def text_to_chunks(text, chunk_size, chunk_overlap):
    words = text.split()
    total_words = len(words)

    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap values")

    chunks = []
    i = 0

    while i < total_words:
        start = i
        end = min(start + chunk_size, total_words)
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        i += (chunk_size-chunk_overlap)

    return chunks

def get_embedding(text, client,model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   time.sleep(20)
   return client.embeddings.create(input = [text], model=model).data[0].embedding



urls = ["https://en.wikipedia.org/wiki/Elon_Musk", "https://en.wikipedia.org/wiki/Bill_Gates","https://en.wikipedia.org/wiki/Sam_Altman"]
text = ""
for url in urls:
    ex_text = extract_text_from_url(url)
    text += ex_text

chunk_size = 750
chunk_overlap = 150
chunks = text_to_chunks(text,chunk_size,chunk_overlap)

df = pd.DataFrame(columns=['Embedding', 'Document'])
load_dotenv()
openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))


for c in chunks:
    embedding = get_embedding(c,openai_client)
    new_row = pd.DataFrame({'Embedding': [embedding], 'Document': [c]})
    df = pd.concat([df, new_row], ignore_index=True)







csv_file_path = 'embeddings.csv'
df.to_csv(csv_file_path, index=False)
