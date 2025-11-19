# ADS Lab Assignment 2

This repository contains the code for Assignment 2 of the Advanced Database Systems (ADS) Lab.

## Files

- `main.py`  
  -Performs **Hybrid Search** on NIPS 2024 papers:  
  -Scrapes papers (title, authors, link)  
  -Stores them in MongoDB  
  -Computes embeddings for titles using `SentenceTransformer`  
  -Allows searching papers by query keywords

- `reverse_image_search.py`  
  -Performs **Reverse Image Search**:  
  -Computes embeddings of 20–30 images using ViT model  
  -Finds the top 3 similar images to a query image based on cosine similarity




