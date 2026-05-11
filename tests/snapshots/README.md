# Pipeline snapshot

The file `pipeline_snapshot.json` is the golden output of
`StandRagPipeline.prepare()` for a fixed question, with a deterministic fake
LLM whose responses are recorded in
`../fixtures/llm_responses/responses.json`.

## Recording the snapshot (one-time, requires live vLLM + stand resources)

```bash
# 1. Make sure resources/stand_nsu/ is populated (see scripts/download_knowledge.py).
# 2. Make sure VLLM_ENDPOINTS points to a live vLLM with the target model loaded.
# 3. Run the recording script (committed temporarily; remove after recording):
uv run python scratch_record_snapshot.py
```

The script lives in plan Task 8, Step 4. After recording, commit
`tests/fixtures/llm_responses/responses.json` and
`tests/snapshots/pipeline_snapshot.json` and delete the scratch script.

If neither file is populated, the test skips with a clear message.
