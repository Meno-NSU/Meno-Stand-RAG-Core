"""FRIDA embeddings + FAISS index for the chunk dataset, in Meno-Stand-RAG-Core format.
Encodes "search_document: " + index_text of indexed chunks (global order of chunk_mapping), CLS pooling, L2-normalised,
T5EncoderModel ai-forever/FRIDA, max 512 tokens (same as meno_rag.stand.search). Writes faiss_frida.index (IndexFlatL2 over
normalised vectors: RAG-Core turns the returned L2 distance into a score) and embeddings shards (float16 .npy, resumable).
embed_frida.py <chunks_dir> [--device cpu|cuda] [--batch 32] [--threads N] [--limit N] [--shard-size 20000]"""
import glob, json, os, sys, time
import numpy as np
K = "/workspace/data/meno-kb-20261101"
os.environ.setdefault("HF_HOME", f"{K}/work/hf_home")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
CH = sys.argv[1]
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
DEV, BS, TH, LIM, SH = arg("--device", "cpu"), arg("--batch", 32), arg("--threads", 32), arg("--limit", 0), arg("--shard-size", 20000)
PI, PN = map(int, arg("--part", "0/1").split("/"))
if DEV == "cpu": torch.set_num_threads(TH)
if os.path.exists(f"{CH}/chunks.jsonl"):
    texts = [json.loads(l)["index_text"] for l in open(f"{CH}/chunks.jsonl") if '"indexed": true' in l]
else:  # release layout: chunks/part-*.jsonl.gz, indexed chunks in part order = global chunk order of the ragcore mapping
    import gzip
    texts = [c["index_text"] for f in sorted(glob.glob(f"{CH}/chunks/part-*.jsonl.gz")) for c in map(json.loads, gzip.open(f, "rt")) if c["indexed"]]
if LIM: texts = texts[:LIM]
out = f"{CH}/ragcore/knowledge/frida_emb"; os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained("ai-forever/FRIDA")
model = T5EncoderModel.from_pretrained("ai-forever/FRIDA", torch_dtype=torch.bfloat16 if DEV == "cpu" else torch.float16).to(DEV).eval()
t0 = time.time()
for s0 in range(0, len(texts), SH):
    fn = f"{out}/emb_{s0 // SH:05d}.npy"
    if os.path.exists(fn) or (s0 // SH) % PN != PI: continue
    part = texts[s0:s0 + SH]
    order = sorted(range(len(part)), key=lambda i: len(part[i]))  # length bucketing, less padding
    E = np.zeros((len(part), 1536), dtype=np.float16)
    with torch.inference_mode():
        for b in range(0, len(order), BS):
            idx = order[b:b + BS]
            x = tok(["search_document: " + part[i] for i in idx], max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEV)
            h = model(**x).last_hidden_state[:, 0].float()
            E[idx] = F.normalize(h, p=2, dim=1).cpu().numpy().astype(np.float16)
    np.save(fn + ".tmp.npy", E); os.replace(fn + ".tmp.npy", fn)
    done = min(s0 + SH, len(texts)); el = time.time() - t0
    print(time.strftime("%H:%M:%S"), "done", done, "/", len(texts), f"{done / max(el, 1):.1f}/s", flush=True)
shards = sorted(f for f in glob.glob(f"{out}/emb_*.npy") if ".tmp" not in f)
if sum(np.load(f, mmap_mode="r").shape[0] for f in shards) == len(texts):
    import faiss
    X = np.concatenate([np.load(f).astype(np.float32) for f in shards])
    idx = faiss.IndexFlatL2(X.shape[1]); idx.add(X)
    faiss.write_index(idx, f"{CH}/ragcore/knowledge/faiss_frida.index")
    print("faiss index", idx.ntotal, "vectors written")
