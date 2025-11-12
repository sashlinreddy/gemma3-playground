# Getting Started

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Add your API key to `.env`:
   ```
   GEMINI_API_KEY=your-key
   ```
3. Drop the PDF corpus into `data/raw/`.
4. Run the ingestion and vectorization steps (see respective docs).
