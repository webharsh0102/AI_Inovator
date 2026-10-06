# Super Investing · AI Innovator Assignment · Part B

**Time:** about 45 minutes. **AI:** obviously allowed. Use whatever models, tools and coding assistants you like.

## The task

Super Investing helps long-term investors research Indian stocks. Build a small **AI research agent**:

> **Input:** an NSE ticker, plus a set of documents about the company
> **Output:** a short **research brief** in Markdown

The brief should have these sections:

1. **Snapshot:** what the company does and its latest results, in 3–4 lines
2. **Bull case:** the strongest reasons to be positive
3. **Bear case:** the strongest reasons to be cautious
4. **Open questions:** what's unclear, missing or conflicting in the sources
5. **Sources:** every claim in the brief should be traceable to a source

Keep it to about one page. It's written for a retail investor, not an analyst.

## The test case (required)

`research_pack/` has 8 documents about **Sarvottam Cables Ltd (NSE: SRVCABLE)**. The company is **fictional**; don't search for it online.

Treat the folder as what a web scraper returned for this ticker. Each file starts with its source, URL and publication date. Assume **today is 23 September 2026**.

## Build it however you like

- A Python or JS script calling any LLM API, an agent framework (LangGraph, CrewAI, OpenAI Agents SDK, Claude Agent SDK…), or a no-code workflow (n8n, Claude Projects/skills, custom GPTs, Dify…). All are fine.
- Free tiers are enough. You don't need to spend money.
- **Bonus, not required:** add a live-data tool (web search, news API, NSE data) and run the agent on 1–2 real NSE tickers.

## What to submit

1. **Link to your build:** a GitHub repo, or for no-code, a shared link or exported workflow file. Must include:
   - Your **system prompt or skill file**, as a separate file
   - A short README: how to run it, and which model(s) you used and why
2. **The brief your agent produced for SRVCABLE**, pasted into the form exactly as generated. If you edited it by hand, say what you changed.
3. **Test log:** run the agent **at least 3 times**, changing your prompt or design between runs. Tell us what went wrong and what you changed.
4. **A screen recording of 2–3 minutes** (Loom, Drive or unlisted YouTube): show it running, and walk through one design decision you're proud of and one limitation.

We care more about **how your agent thinks** than about UI or code polish.
