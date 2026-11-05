# KB release tools

Tools that turn the source layer of the dataset [Meno-NSU/NSU-Data-For-Meno](https://huggingface.co/datasets/Meno-NSU/NSU-Data-For-Meno)
into a KB release (`kb/chunks/v1`): structure-aware chunking (`chunk_kb.py`), the BM25 index (`rag_bm25.py`), FRIDA
embeddings (`embed_frida.py`) and their helpers. A build copies this directory into the release it makes
(`kb/chunks/v1/tools`), so every release carries the exact code that chunked it.

Origin: the dataset revision `76fa38bd10853e0deaa686b945e6397b1f51eb94`, path `kb/chunks/v1/tools`
(sha256 of `chunk_kb.py` there: `54a7006496b52f6e18ef0675d4bb10f3b70fbe83b9f95eb15da77ef3ae733e03`).
Changes made here are listed in the docstring of `chunk_kb.py` and marked `FIX(meno)` in the code.
The files are kept close to upstream (not reformatted), so they are excluded from ruff.

`builds/` holds the scripts that made the current KB releases from a dataset snapshot: `build_v29.py` (source layer v2.9 +
the selected social crawl, social sites relabelled, FIT review cards normalised; one chunker pass, then BM25) and
`build_handbook.py` (a derived release: an existing one + the NSU handbook for Menon, one document per section). Both take
this directory as the chunker (`TOOLS`) and copy it into the release they make.
