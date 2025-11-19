import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from urllib.parse import quote_plus
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from urllib.parse import quote_plus

username = "fa23bcs181"
password = "pass@1234"  

password = quote_plus(password)
uri = f"mongodb+srv://{username}:{password}@cluster0.wtckmr6.mongodb.net/?appName=Cluster0"


client = MongoClient(uri)
db = client["hybrid_search"]
papers = db["nips_papers"]

print("Connected to MongoDB successfully!")

#------------------scrapping-------------
url = "https://papers.nips.cc/paper_files/paper/2024"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

paper_elements = soup.find_all("li")
print(f"Found {len(paper_elements)} list items.")

paper_list = []

for li in paper_elements:
    a_tag = li.find("a")
    if not a_tag:
        continue  


    title = a_tag.text.strip()
    
    
    link = "https://papers.nips.cc" + a_tag['href']

    
    authors_text = li.text.replace(title, "").strip()
    authors = [author.strip() for author in authors_text.split(",") if author.strip()]

    paper_list.append({
        "title": title,
        "authors": authors,
        "link": link
    })

print(f"Extracted {len(paper_list)} papers.")
for paper in paper_list[:5]:
    print(paper)


valid_papers = [paper for paper in paper_list if paper["title"]]
print(f"Valid papers to insert: {len(valid_papers)}")


# Insert into MongoDB
inserted_count = 0
for paper in valid_papers:
    papers.update_one(
        {"title": paper["title"]},  # check if already exists
        {"$set": paper},
        upsert=True
    )
    inserted_count += 1

print(f"Inserted/Updated {inserted_count} papers into MongoDB!")


model = SentenceTransformer('all-MiniLM-L6-v2')  

for paper in valid_papers:
    embedding = model.encode(paper["title"]).tolist()  # convert to list for MongoDB
    papers.update_one(
        {"title": paper["title"]},
        {"$set": {"embedding": embedding}}
    )

print("Added embeddings for all papers!")

#---------search query----
query = "Graph Neural Networks"  
query_vec = model.encode(query)

from numpy import dot
from numpy.linalg import norm

results = []
for paper in papers.find():
    emb = paper.get("embedding")
    if emb:
        sim = dot(query_vec, emb) / (norm(query_vec) * norm(emb))  # cosine similarity
        results.append((sim, paper))


results = sorted(results, key=lambda x: x[0], reverse=True)
for score, paper in results[:5]:
    print(score, paper["title"])

