import os
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTModel
from numpy import dot
from numpy.linalg import norm

IMAGE_FOLDER = "images"
QUERY_IMAGE = "greenery.png"  


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

def get_image_embedding(image_path):
    """Load an image and return its embedding vector."""
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding


#load all images &compute embeddings
image_embeddings = {}
for img_file in os.listdir(IMAGE_FOLDER):
    if img_file.endswith(".png") and img_file != os.path.basename(QUERY_IMAGE):
        path = os.path.join(IMAGE_FOLDER, img_file)
        emb = get_image_embedding(path)
        image_embeddings[img_file] = emb
print(f"Computed embeddings for {len(image_embeddings)} images.")


#compute embedding for query image
query_embedding = get_image_embedding(QUERY_IMAGE)

#----------cosine similarity-----
results = []
for img_name, emb in image_embeddings.items():
    sim = dot(query_embedding, emb) / (norm(query_embedding) * norm(emb))
    results.append((sim, img_name))



results = sorted(results, key=lambda x: x[0], reverse=True)
print("Top 3 similar images to the {QUERY_IMAGE}:")
for sim, img_name in results[:3]:
    print(f"{img_name} with similarity {sim:.4f}")

