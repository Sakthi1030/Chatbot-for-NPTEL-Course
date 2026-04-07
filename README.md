# Chatbot for NPTEL Course

This project is now a simple Streamlit RAG app for course documents.

## What changed

- Removed the old Colab-only workflow.
- Replaced GPT-2 local generation with the Gemini Developer API.
- Added local document loading for PDF, TXT, MD, and DOCX files.
- Added retrieval over chunked documents using TF-IDF, then answer generation with Gemini.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file from `.env.example` and add your Gemini API key.
4. Start the app:

```bash
streamlit run app.py
```

## Deploy to Render

This repo includes a `render.yaml` blueprint for a Render web service.

1. Push this repo to GitHub.
2. In Render, choose `New` -> `Blueprint` or `New` -> `Web Service`.
3. Set the secret environment variable `GEMINI_API_KEY`.
4. Optionally set `DOCS_DIR` if your documents are bundled into the repo.

The service starts with:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Important: Render free web services currently spin down after 15 minutes of inactivity, so use a paid instance if you want it to stay warm all the time.

If `GDRIVE_FOLDER_URL` is set, the app will try to download documents from that public Google Drive folder into `DOCS_DIR` on startup.

## Document folder

Set `DOCS_DIR` in `.env` or paste the folder path into the app. This can be:

- A normal local folder
- A Google Drive desktop sync folder
- Any mounted Drive path available on your machine
- A `docs/` folder committed into this repo for cloud deployment

## Render note about documents

If you deploy to Render, the app cannot read files directly from your personal Google Drive on your laptop. For Render hosting, your documents need to be:

- Committed into the repo, such as a `docs/` folder
- Downloaded from a public Google Drive folder through `GDRIVE_FOLDER_URL`
- Stored in an external database or object store

## Suggested free model

The app defaults to `gemini-3-flash-preview`. If preview availability changes for your account, try `gemini-2.5-flash`.
