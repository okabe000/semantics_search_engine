# Technical Overview

This document complements the main README by unpacking the architectural choices, data flow, and extensibility points of the multi-stage semantic search project.

## System Architecture
- **Client:** `frontend/index.html` (vanilla JS). Renders stage cards, manages drag-and-drop ordering, assembles payloads, and streams gallery updates.
- **API Layer:** `api.py` (FastAPI). Exposes search and ingestion endpoints, handles preprocessing, and coordinates with the vector database.
- **Vector Store:** Qdrant (embedded, file-backed). Stores CLIP embeddings and optional face embeddings; exposes cosine-similarity search.
- **Models:**
  - `clip-ViT-B-32` SentenceTransformer for all text ↔ image embeddings.
  - DeepFace + `Facenet512` for 512-d face embeddings, with fallbacks across `retinaface`, `mtcnn`, `opencv`, `ssd` detectors.
- **Storage Helpers:** `loaded_images.json` tracks processed files so batches can skip duplicates; images served via FastAPI static mounts (`/images`, `/unstracted`).

```
Frontend UI → FastAPI → CLIP / DeepFace → Qdrant → Ranked Results → Frontend Gallery
```

## Multi-Stage Search Flow
1. **Instructive Stage (`stage_type="instructive"`):**
   - Optional text prompt and reference image.
   - Generates a combined embedding (prompt + image) to seed search.
2. **Example Stage (`stage_type="examples"`):**
   - `like_images` and `not_like_images` arrays (base64 from the UI).
   - Computes centroid vectors for positives/negatives before querying Qdrant again.
3. **Faces Stage (`stage_type="faces"`):**
   - Optional include/exclude face lists and `require_all_faces` flag.
   - Uses DeepFace embeddings and cosine/euclidean thresholds (`face_match_threshold`, default 0.4).

Each stage consumes the previous stage’s shortlist; if no prior results exist, the backend falls back to a broad Qdrant scroll.

## Key Algorithms
- **Vector Similarity:** Cosine similarity for CLIP embeddings (`Distance.COSINE` in Qdrant).
- **Face Matching:** Normalized embeddings compared with both cosine similarity (`>` threshold) and Euclidean distance (`< 0.9`).
- **Batch Ingestion:** Thread pool across folders, skipping already processed files via `loaded_images.json`.
- **Lazy Gallery Loading:** IntersectionObserver in the frontend defers image fetches until tiles are in view.

## API Reference
| Endpoint | Method | Description |
| --- | --- | --- |
| `/multi_stage_search` | POST | Executes the ordered pipeline; returns ranked payloads with optional similarity scores. |
| `/all` | GET | Returns up to 1,000 stored image payloads for gallery population. |
| `/stats` | GET | Summaries (image count, vector size, face-model enablement, detection backends, face counts). |
| `/reload_images` | POST | Batch process images from a folder (`unstracted` by default). |
| `/reset_tracking` | DELETE | Clears `loaded_images.json` and forces reprocessing on next ingestion. |

## Configuration Notes
- **Model Downloads:** SentenceTransformers and DeepFace will download weights the first time they run. Cache them before demoing offline.
- **Face Recognition Toggle:** If DeepFace fails to import, the API auto-disables face filtering and communicates the status via `/stats`.
- **Qdrant Persistence:** Data lives in `./qdrant_storage`. For production, switch to a managed Qdrant instance and update the client connection.
- **Environment Variables:** Not required today, but you can externalize paths, thresholds, and CORS rules with minimal editing in `api.py`.

## Extending the Project
- **New Stage Types:** Add a content builder in the frontend, define a new `stage_type`, and implement a handler in `api.py` similar to `process_examples_stage`.
- **Custom Ranking:** Introduce re-ranking (e.g., CLIP-score weighting or temporal decay) before returning results in `format_results`.
- **Analytics Hooks:** Log stage payloads and result metrics for A/B testing different stage orders or thresholds.
- **Authentication:** Layer FastAPI dependencies or API keys to protect ingestion endpoints when moving beyond local demos.

## Testing & Troubleshooting
- **Local Smoke Test:** After starting the API, hit `http://localhost:8000/docs` to try `/all` and `/stats` quickly.
- **Face Debugging:** Use `/stats` to ensure DeepFace is active. If disabled, confirm dependencies (`deepface`, `opencv-python-headless`, `tf-keras`) are installed.
- **Data Refresh:** Use `/reload_images` if newly added files aren’t appearing; call `/reset_tracking` for a full rebuild.

---

For a narrative overview suitable for non-technical stakeholders, refer to the root `README.md`. This file is meant for engineers who want to understand how things fit together under the hood.
