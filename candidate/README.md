# Source-grounded NSE research brief (RAG)

A small CLI that reads the supplied Markdown research pack and writes a one-page, citation-backed research brief. It does not browse the web. The documents are treated as untrusted input; instructions embedded in source material are ignored.

## Run

Python 3.10+ is recommended. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Set HF_TOKEN, HF_PROVIDER, and HF_MODEL in .env.
python src\research_agent.py SRVCABLE --docs research_pack --top-k 8 --dry-run
# Review the selected evidence before making a hosted inference request.
python src\research_agent.py SRVCABLE --docs research_pack --top-k 8 --generate
```

After generation, the brief opens in a desktop window with colored section headings, highlighted citations, and clickable source links. The window uses Python's built-in Tkinter library.

Brief generation uses Hugging Face hosted Inference Providers with `InferenceClient`; the generation model is not downloaded locally. Set `HF_TOKEN` to an inference-enabled Hugging Face token, `HF_PROVIDER` to a provider enabled for your account, and `HF_MODEL` to a chat-completion model that provider serves. The example configuration uses provider `nscale` and model `Qwen/Qwen2.5-Coder-7B-Instruct`. The `LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-GGUF` repository contains GGUF model files; a token alone cannot serve those files as an inference API. Use a model shown as available for chat completion in Hugging Face Inference Providers, or deploy the GGUF separately as an Inference Endpoint. Retrieval still uses local sentence-transformer embedding and reranker models, downloaded from Hugging Face on first use.

`--dry-run` executes ingestion, local retrieval and prompt/configuration checks, then prints the selected sources and a rough input-token estimate without calling hosted inference. Run `--generate` only after reviewing that output. If Hugging Face returns `model_not_supported`, choose a chat model with a provider enabled for your account; merely changing the repository ID to another unsupported model will not resolve it.

## Retrieval design

Markdown front matter is parsed into source, URL, date and type metadata. Chunks are built at paragraph boundaries (up to 700 words), preserving tables and adjacent evidence rather than slicing fixed character windows. Oversized paragraphs use 600-word windows with 80-word overlap. Retrieval combines BM25 and normalized embedding similarity using reciprocal-rank fusion, then reranks a broad candidate pool with a cross-encoder. Top-k is configurable (default 8) so both primary filings and material counter-evidence can reach synthesis. Citations are assigned to retrieved chunks and must be used by the model. The prompt prioritizes official filings, calls out conflicts, separates company claims from established facts, and prohibits investment advice.

## Known pack issues to inspect

The company release reports Q1 revenue of ₹1,248 crore; Business Daily reports ₹1,428 crore. The discrepancy should be surfaced and checked against the official release. The 2024 pledge story is stale relative to the June 2026 exchange filing. The Nagpur cable-TV penalty concerns a different entity. The promotional blog contains an instruction-injection attempt and unsupported buy/target claims. The README's assumed date is 23 September 2026, while the environment may be later; this agent uses source publication dates and does not use system date to infer freshness.

## Retrieval evaluation (offline)

The loader excludes the assignment README and ingests eight evidence chunks. Three local retrieval comparisons were run with the test query:

1. **BM25 baseline:** official Q1 release ranked first; then September GST filing, old pledge report, promotional blog and call transcript.
2. **Dense embedding baseline:** shareholding filing ranked first, followed by the official release, blog, GST filing and old pledge report. Semantic similarity alone overweights broad governance terms.
3. **Hybrid RRF + cross-encoder (current):** official Q1 release ranked first; then blog, stale pledge report, call transcript, GST filing, conflicting Business Daily article, shareholding, and unrelated Nagpur cable-TV article. Reranking improved primary-results placement but exposed a limitation: the unrelated entity can still enter when `TOP_K=8`.

Design changes: combine lexical and semantic rankings using reciprocal-rank fusion, rerank a broad candidate pool, and pass source type/date/URL with every chunk. Keep top-k configurable; on this small pack, `TOP_K=6` excludes the unrelated article while retaining major company disclosures and the revenue conflict. The prompt instructs the model to resolve entity identity and ignore source-embedded instructions.

These are retrieval-only runs. End-to-end hosted generation should be run with `--generate` and a valid `HF_TOKEN` and supported `HF_MODEL`.

## End-to-end run log

Run the CLI three times while varying `TOP_K` (6 vs 8) and the system prompt, record each generated brief and failure, then select the best evidenced one-page brief. The required screencast should show one retrieval decision and the entity-resolution limitation.
