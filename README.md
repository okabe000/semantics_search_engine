# Multi-Stage Semantic Search

A portfolio-ready project that lets you explore large image libraries the way you think—by combining natural language, example images, and even specific faces in one guided search.

## At a Glance
- Search visually rich collections without perfect keywords.
- Mix and match three search styles (text, examples, faces) in a drag-and-drop pipeline.
- See instant feedback thanks to a lightweight FastAPI backend plus a responsive vanilla JS frontend.

## What You Can Do
- **Creative teams:** find the right shot faster by blending art direction prompts with “looks like this / not like this” references.
- **Knowledge managers:** enforce brand or compliance rules by excluding certain faces or looks.
- **Investigators & analysts:** sift through surveillance or archival imagery with precise face filtering.

## Quick Start
1. **Install dependencies**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Launch the API**
   ```bash
   uvicorn api:app --reload
   ```
3. **Open the UI**  
   Double-click `frontend/index.html` (or serve it via any static host) and point the interface at your running API (`http://localhost:8000` by default).

## Project Tour
- `api.py` &mdash; FastAPI service that:
  - encodes text and images with the `clip-ViT-B-32` SentenceTransformer,
  - stores vectors in a local Qdrant collection for fast similarity search,
  - optionally extracts face embeddings with DeepFace (`Facenet512`) so you can include / exclude people.
- `frontend/` &mdash; a single-page, dependency-free interface with staged search cards, drag-and-drop reordering, and lazy-loaded image galleries.
- `docs/TECHNICAL_OVERVIEW.md` &mdash; supplemental architecture notes when you want the nitty-gritty.

## Tech Stack & Models
- **Backend:** FastAPI, Python 3, Pillow, NumPy, ThreadPoolExecutor for batch ingestion.
- **Vector Intelligence:** SentenceTransformers `clip-ViT-B-32` for joint text–image embeddings.
- **Vector Database:** Qdrant (local disk storage in this setup).
- **Face Recognition (optional):** DeepFace with the `Facenet512` embedding model plus multi-backend detectors (`retinaface`, `mtcnn`, `opencv`, `ssd`).
- **Frontend:** Vanilla JS/CSS/HTML with IntersectionObserver for smooth, lazy gallery loading.

## Use Cases to Highlight 
- Build an *“AI photo librarian”* that understands creative briefs and subjective style.
- Create a *brand safety guardrail* that spots disallowed imagery or people before publishing.
- Support *investigative review* workflows that need both concept and face matching in one query.

<details>
<summary><strong>Technical Deep Dive (Optional)</strong></summary>

### Multi-Stage Search Pipeline
1. **Instructive stage** — Combines a natural-language prompt with an optional reference image to seed results.
2. **Example stage** — Refines the candidate set by pulling images similar to positives and away from negatives.
3. **Faces stage** — Adds include / exclude face lists, with an option to require that all selected faces appear.

Each stage works on the result set from the previous stage, so ordering matters. Dragging cards in the UI instantly reorders the pipeline.

### Algorithms & Data Flow
- Text/image prompts are embedded with CLIP and queried against Qdrant using cosine similarity.
- Example-based refinement re-scores vectors by averaging positives and subtracting negatives before querying.
- Face filtering extracts normalized 512-d embeddings via DeepFace, then performs cosine + Euclidean similarity checks to keep matches under configurable thresholds.
- Processed images are tracked in `loaded_images.json` to avoid recomputation; batch ingestion uses a thread pool for throughput.

### API Surface
- `POST /multi_stage_search` — orchestrates the staged query payload from the frontend.
- `GET /all` — paginates the gallery view.
- `GET /stats` — returns collection size, vector dimensionality, face-model status, and face counts.
- `POST /reload_images` / `DELETE /reset_tracking` — maintenance endpoints for ingestion workflows.

</details>

## Next Ideas
- Expand stages (e.g., color palette, composition heuristics).
- Persist user-defined pipelines for different teams.
- Swap in hosted Qdrant / managed vector DB and CLIP variants for scale.

---

🎯 **Portfolio tip:** Snapshot a short walkthrough video showing the three-stage refinement in action and link it beside this repo on your LinkedIn profile. It conveys the core value in under a minute.
