# Deploying to Google Cloud Run

Everything below uses placeholders - replace `YOUR_PROJECT_ID`, `YOUR_REGION` (e.g.
`europe-west1`) and `YOUR_REPO`. Nothing in this repository contains real project ids or keys.

## Architecture on GCP

- **Cloud Run `eu-ai-act-rag-api`** - FastAPI container (`docker/api.Dockerfile`), runs as a
  dedicated service account; Gemini and Vertex embeddings / ranking are called with that
  identity (no API key).
- **Cloud Run `eu-ai-act-rag-ui`** - nginx serving the static build, proxying `/api` to the API.
- **Cloud SQL for PostgreSQL + pgvector** (optional) - set `VECTOR_STORE=pgvector`; otherwise the
  in-memory store is rebuilt at start-up (fine for the ~3k-chunk corpus, ~seconds with cached
  embeddings).
- **Secret Manager** - only for third-party keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `COHERE_API_KEY`) and the database URL. Mounted as env vars, never baked into images.

## 1. One-time setup

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  aiplatform.googleapis.com secretmanager.googleapis.com discoveryengine.googleapis.com

gcloud artifacts repositories create YOUR_REPO --repository-format=docker --location=YOUR_REGION

# least-privilege runtime identity for the API
gcloud iam service-accounts create rag-api --display-name="eu-ai-act-rag API"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:rag-api@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
# only if RERANKER=vertex
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:rag-api@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/discoveryengine.viewer"
```

## 2. Secrets (only those you use)

```bash
printf '%s' "your-key-here" | gcloud secrets create anthropic-api-key --data-file=-
gcloud secrets add-iam-policy-binding anthropic-api-key \
  --member="serviceAccount:rag-api@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## 3. Build and deploy

```bash
gcloud builds submit --config deploy/cloudrun/cloudbuild.yaml \
  --substitutions=_REGION=YOUR_REGION,_REPO=YOUR_REPO .
```

or manually:

```bash
IMAGE=YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO/eu-ai-act-rag-api:latest
docker build -f docker/api.Dockerfile -t $IMAGE . && docker push $IMAGE

gcloud run deploy eu-ai-act-rag-api --image=$IMAGE --region=YOUR_REGION \
  --service-account=rag-api@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --cpu=1 --memory=1Gi --min-instances=0 --max-instances=3 --concurrency=20 --timeout=300 \
  --set-env-vars=LLM_PROVIDER=gemini,EMBEDDING_PROVIDER=vertex,RETRIEVER=hybrid,GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,GOOGLE_CLOUD_LOCATION=YOUR_REGION \
  --set-secrets=ANTHROPIC_API_KEY=anthropic-api-key:latest \
  --no-allow-unauthenticated
```

`service.yaml` is the declarative equivalent (`gcloud run services replace deploy/cloudrun/service.yaml`).

## Notes and trade-offs

- **Corpus in the image vs. at start-up**: the image ships the fixture only. For production,
  run `euai ingest` in a Cloud Run job writing `corpus.jsonl` to a GCS bucket mounted with a
  Cloud Storage volume, and set `CORPUS_PATH` to the mount path.
- **Cold starts**: with `EMBEDDING_PROVIDER=vertex` the index is rebuilt at start-up unless
  pgvector is used; prefer pgvector (Cloud SQL) or `--min-instances=1` for latency-sensitive use.
- **Auth**: the API is deployed `--no-allow-unauthenticated`; put the UI behind IAP or an API
  gateway rather than exposing the API publicly.
- The cross-encoder (`RERANKER=local`) needs the `[local]` extra (torch) - a much larger image
  and ~2 GiB memory; `RERANKER=vertex` keeps the container slim.
