from __future__ import annotations
import argparse, os, re, sys
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import InferenceClient

@dataclass
class Chunk:
    sid: str
    text: str
    title: str
    source: str
    url: str
    date: str
    kind: str


def read_docs(folder: Path) -> list[Chunk]:
    chunks=[]
    for path in sorted(folder.glob('*.md')):
        if path.name.lower() == 'readme.md':
            continue
        raw=path.read_text(encoding='utf-8-sig')
        parts=raw.split('---',2)
        meta={}
        if len(parts)>=3:
            for line in parts[1].splitlines():
                if ':' in line:
                    k,v=line.split(':',1); meta[k.strip().lower()]=v.strip()
            body=parts[2].strip()
        else: body=raw
        title=next((x.lstrip('# ').strip() for x in body.splitlines() if x.startswith('# ')),path.stem)
        # Chunk by paragraph, keeping tables/list rows intact. Merge short adjacent paragraphs
        # up to ~700 words; split only oversized paragraphs with overlap.
        paras=[p.strip() for p in re.split(r'\n\s*\n',body) if p.strip()]
        pieces=[]; buf=''
        for p in paras:
            if len((buf+'\n\n'+p).split())<=700:
                buf=(buf+'\n\n'+p).strip()
            else:
                if buf: pieces.append(buf)
                if len(p.split())<=700: buf=p
                else:
                    words=p.split()
                    for i in range(0,len(words),600): pieces.append(' '.join(words[max(0,i-80):i+600]))
                    buf=''
        if buf: pieces.append(buf)
        for i,text in enumerate(pieces,1):
            chunks.append(Chunk(f'{path.stem}:{i}',text,title,meta.get('source',path.stem),meta.get('url',''),meta.get('published','unknown'),meta.get('type','unknown')))
    return chunks


def tokens(s): return re.findall(r'[a-z0-9]+',s.lower())

def retrieve(query, docs, encoder, reranker, top_k=8):
    corpus=[tokens(d.text+' '+d.title+' '+d.source) for d in docs]
    bm=BM25Okapi(corpus)
    lexical=np.asarray(bm.get_scores(tokens(query)))
    embs=encoder.encode([d.text+' '+d.title for d in docs],normalize_embeddings=True)
    q=encoder.encode([query],normalize_embeddings=True)[0]
    semantic=embs@q
    # Reciprocal-rank fusion avoids scale mismatch; retrieve a broad candidate pool first.
    order_b=np.argsort(-lexical); order_s=np.argsort(-semantic); fused={}
    for order in (order_b,order_s):
        for rank,idx in enumerate(order[:max(top_k*4,20)]): fused[int(idx)]=fused.get(int(idx),0)+1/(60+rank+1)
    cand=sorted(fused,key=fused.get,reverse=True)[:max(top_k*3,16)]
    scores=reranker.predict([(query,docs[i].text) for i in cand])
    ranked=sorted(zip(cand,scores),key=lambda x:float(x[1]),reverse=True)
    return [docs[i] for i,_ in ranked[:top_k]]


def generate_brief(model_name, provider, token, instructions, user_input):
    client = InferenceClient(model=model_name, provider=provider, token=token)
    response = client.chat_completion(
        messages=[
            {'role': 'system', 'content': instructions},
            {'role': 'user', 'content': user_input},
        ],
        max_tokens=1200,
        temperature=0,
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('Hugging Face returned an empty or non-text response.')
    return content


def show_brief_gui(brief: str, ticker: str) -> None:
    """Display the generated Markdown brief in a simple, colored desktop window."""
    import re
    import tkinter as tk
    import webbrowser
    from tkinter import ttk

    root = tk.Tk()
    root.title(f'{ticker} Research Brief')
    root.geometry('900x720')
    root.minsize(640, 480)
    root.configure(bg='#f3f6fb')

    style = ttk.Style(root)
    style.configure('Brief.TFrame', background='#f3f6fb')
    style.configure('Title.TLabel', background='#f3f6fb', foreground='#17365d',
                    font=('Segoe UI', 18, 'bold'))
    style.configure('Hint.TLabel', background='#f3f6fb', foreground='#64748b',
                    font=('Segoe UI', 9))

    frame = ttk.Frame(root, padding=18, style='Brief.TFrame')
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text=f'{ticker} Research Brief', style='Title.TLabel').pack(anchor='w')
    ttk.Label(frame, text='Source-grounded summary', style='Hint.TLabel').pack(anchor='w', pady=(2, 12))

    body = ttk.Frame(frame)
    body.pack(fill='both', expand=True)
    scroll = ttk.Scrollbar(body)
    scroll.pack(side='right', fill='y')
    view = tk.Text(body, wrap='word', yscrollcommand=scroll.set, padx=18, pady=14,
                   bg='white', fg='#243247', relief='flat', borderwidth=0,
                   font=('Segoe UI', 11), spacing1=2, spacing3=7)
    view.pack(side='left', fill='both', expand=True)
    scroll.configure(command=view.yview)
    view.tag_configure('h1', font=('Segoe UI', 15, 'bold'), foreground='#17365d', spacing1=12, spacing3=8)
    view.tag_configure('h2', font=('Segoe UI', 12, 'bold'), foreground='#2563a6', spacing1=9, spacing3=4)
    view.tag_configure('bullet', foreground='#2563a6', font=('Segoe UI', 11, 'bold'))
    view.tag_configure('citation', foreground='#16805d', font=('Segoe UI', 10, 'bold'))
    view.tag_configure('link', foreground='#1769aa', underline=True)

    def insert_inline(text: str) -> None:
        # Render Markdown links as clickable labels while keeping their visible title.
        cursor = 0
        pattern = r'\[([^\]]+)\]\((https?://[^)]+)\)|(\[S\d+(?:[, ]+S\d+)*\])'
        for match in re.finditer(pattern, text):
            view.insert('end', text[cursor:match.start()])
            if match.group(2):
                start = view.index('end-1c')
                view.insert('end', match.group(1), 'link')
                end = view.index('end-1c')
                tag = f'url_{start.replace(".", "_")}'
                view.tag_add(tag, start, end)
                view.tag_bind(tag, '<Button-1>', lambda _event, url=match.group(2): webbrowser.open(url))
            else:
                view.insert('end', match.group(3), 'citation')
            cursor = match.end()
        view.insert('end', text[cursor:])

    for line in brief.splitlines():
        stripped = line.strip()
        if not stripped:
            view.insert('end', '\n')
        elif stripped.startswith('# '):
            view.insert('end', stripped[2:] + '\n', 'h1')
        elif stripped.startswith('## '):
            view.insert('end', stripped[3:] + '\n', 'h2')
        elif stripped.startswith(('- ', '* ')):
            view.insert('end', '• ', 'bullet')
            insert_inline(stripped[2:])
            view.insert('end', '\n')
        else:
            insert_inline(stripped)
            view.insert('end', '\n')
    view.configure(state='disabled')
    root.mainloop()


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(errors='backslashreplace')
    load_dotenv()
    ap=argparse.ArgumentParser(description='Build a source-grounded company research brief.')
    ap.add_argument('ticker')
    ap.add_argument('--docs',default='research_pack')
    ap.add_argument('--top-k',type=int,default=int(os.getenv('TOP_K','8')))
    ap.add_argument('--dry-run',action='store_true',help='Run loading and retrieval checks without calling hosted inference.')
    ap.add_argument('--generate','--call-api',dest='generate',action='store_true',help='Generate the brief using Hugging Face hosted inference.')
    args=ap.parse_args()
    if args.top_k < 1: raise SystemExit('--top-k must be at least 1')
    if not args.dry_run and not args.generate:
        raise SystemExit('No brief generated. First run with --dry-run; then use --generate to call Hugging Face hosted inference.')
    hf_token=os.getenv('HF_TOKEN','').strip()
    provider=os.getenv('HF_PROVIDER','').strip()
    model=os.getenv('HF_MODEL','').strip()
    if args.generate and not hf_token:
        raise SystemExit('HF_TOKEN is required for hosted inference. Set it in your private .env file.')
    if args.generate and not provider:
        raise SystemExit('HF_PROVIDER is required. Set it to an Inference Provider enabled for your Hugging Face account.')
    if args.generate and not model:
        raise SystemExit('HF_MODEL is required. Set it to a chat-completion model supported by an Inference Provider enabled for your Hugging Face account.')
    print('[1/5] Reading Markdown evidence...',flush=True)
    docs=read_docs(Path(args.docs))
    if not docs: raise SystemExit(f'No Markdown sources found in {args.docs}')
    print(f'      Loaded {len(docs)} evidence chunks from {args.docs}.',flush=True)
    print('[2/5] Loading local embedding and reranking models...',flush=True)
    enc=SentenceTransformer(os.getenv('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2'))
    rerank=CrossEncoder(os.getenv('RERANKER_MODEL','cross-encoder/ms-marco-MiniLM-L-6-v2'))
    query=f'{args.ticker} company overview latest results growth order book margins risks governance legal disputes and unresolved source conflicts'
    print('[3/5] Retrieving and reranking evidence...',flush=True)
    selected=retrieve(query,docs,enc,rerank,args.top_k)
    if not selected: raise SystemExit('Retrieval returned no evidence; refusing to generate a brief.')
    sources=[]
    for i,d in enumerate(selected,1):
        sources.append(f'[S{i}] {d.title}\nPublisher/type: {d.source} ({d.kind}); published {d.date}\nURL: {d.url}\nEvidence: {d.text}')
    prompt=Path(__file__).resolve().parents[1]/'system_prompt.md'
    if not prompt.is_file(): raise SystemExit(f'System prompt not found: {prompt}')
    instructions=prompt.read_text(encoding='utf-8').strip()
    if not instructions: raise SystemExit('System prompt is empty; refusing to generate a brief.')
    evidence_input=f'Ticker: {args.ticker}\n\nEvidence:\n'+'\n\n'.join(sources)
    print('[4/5] Preflight summary (no generation request yet):',flush=True)
    print(f'      Hugging Face model: {model or "(set HF_MODEL before --generate)"}; provider: {provider or "(set HF_PROVIDER before --generate)"}; token configured: {"yes" if bool(hf_token) else "no"}; top-k: {len(selected)}; approximate input size: {(len(instructions)+len(evidence_input)+3)//4} tokens.',flush=True)
    print('      Selected sources:',flush=True)
    for i,d in enumerate(selected,1):
        print(f'        [S{i}] {d.title} | {d.source} | {d.date}',flush=True)
    if args.dry_run:
        print('[5/5] DRY RUN complete. Hugging Face inference was not called.',flush=True)
        return
    print('[5/5] Calling Hugging Face hosted inference...',flush=True)
    brief=generate_brief(model,provider,hf_token,instructions,evidence_input)
    show_brief_gui(brief,args.ticker)

if __name__=='__main__': main()
