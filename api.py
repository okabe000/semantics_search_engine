from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from PIL import Image
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount directories for static files
app.mount("/unstracted", StaticFiles(directory="unstracted"), name="unstracted")
app.mount("/images", StaticFiles(directory="images"), name="images")

# Setup
model = SentenceTransformer("clip-ViT-B-32")
client = QdrantClient(path="./qdrant_storage")
collection_name = "images"

# Get vector size safely (512 for clip-ViT-B-32)
try:
    vector_size = model.get_sentence_embedding_dimension()
    if vector_size is None:
        raise AttributeError
except AttributeError:
    vector_size = model.encode(["test"])[0].shape[0]  # dynamically infer

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=vector_size,
        distance=Distance.COSINE
    )
)
print(f"✅ Vector size set to: {vector_size}")


# --- Load images from unstructured directory ---
def load_images(folder="unstracted"):
    count = 0
    print("🔍 Loading images from unstructured dataset...")
    
    if not os.path.exists(folder):
        print(f"⚠️ {folder} directory not found!")
        return count
        
    print(f"📁 Loading images from {folder}...")
    for file in sorted(os.listdir(folder)):
        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
            
        path = os.path.join(folder, file)
        try:
            img = Image.open(path).convert("RGB")
            vector = model.encode(img)
            
            # Create URL-friendly path
            client.upsert(
                collection_name=collection_name,
                points=[PointStruct(
                    id=count,
                    vector=vector,
                    payload={
                        "path": f"/{folder}/{file}",  # Keep the path relative to root
                        "filename": file
                    }
                )]
            )
            count += 1
            
            if count % 10 == 0:
                print(f"✓ Processed {count} images...")
        except Exception as e:
            print(f"⚠️ Error loading {path}: {str(e)}")
    
    return count

loaded_count = load_images()
print(f"✅ Preloaded {loaded_count} local images into Qdrant.")

# --- API Models ---
class Query(BaseModel):
    text: str

@app.post("/search")
def search(query: Query):
    print(f"Searching for: {query.text}")
    vector = model.encode(query.text)
    print(f"Vector: {vector[:5]}...")  # Print first 5 dimensions for brevity
    # Get top 100 matches with scores, no score threshold (find closest matches even if not exact)
    hits = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=100,
        score_threshold=None  # Return results even if similarity is low
    )
    print(f"Found {len(hits)} hits.")
    # Include similarity scores in response
    return [{"score": hit.score, **hit.payload} for hit in hits]

@app.get("/all")
def all_images():
    results, _ = client.scroll(collection_name=collection_name, limit=100)
    return [r.payload for r in results]
