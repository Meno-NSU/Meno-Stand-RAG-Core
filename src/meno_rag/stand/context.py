from __future__ import annotations

import json
from typing import Any


def normalize_urls(value: str | list[str] | None) -> list[str]:
    """Accept a document ``url`` as a string, a list of strings, or ``None`` and
    return a clean, de-duplicated, order-preserving list of non-empty URLs."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            stripped = item.strip() if isinstance(item, str) else str(item).strip()
            if stripped and stripped not in result:
                result.append(stripped)
        return result
    stripped = str(value).strip()
    return [stripped] if stripped else []


def global_chunk_index_to_local(
    global_chunk_index: int,
    documents: list[dict[str, Any]],
    chunk_mapping: dict[str, dict[str, int]],
) -> tuple[int, int]:
    if str(global_chunk_index) not in chunk_mapping:
        raise ValueError(f"The global chunk index {global_chunk_index} is not supported in the chunk mapping!")
    document_index = chunk_mapping[str(global_chunk_index)]["doc_index"]
    local_chunk_index = chunk_mapping[str(global_chunk_index)]["local_chunk_index"]
    if (document_index < 0) or (document_index >= len(documents)):
        err_msg = f"The chunk mapping is wrong. The global chunk {global_chunk_index} contains a wrong data!\n"
        err_msg += json.dumps(chunk_mapping[str(global_chunk_index)], ensure_ascii=False, indent=4)
        raise ValueError(err_msg)
    if (local_chunk_index < 0) or (local_chunk_index >= len(documents[document_index]["chunks"])):
        err_msg = f"The chunk mapping is wrong. The global chunk {global_chunk_index} contains a wrong data!\n"
        err_msg += json.dumps(chunk_mapping[str(global_chunk_index)], ensure_ascii=False, indent=4)
        raise ValueError(err_msg)
    return document_index, local_chunk_index


def global_chunk_index_to_text(
    global_chunk_index: int, documents: list[dict[str, Any]], chunk_mapping: dict[str, dict[str, int]]
) -> str:
    document_index, local_chunk_index = global_chunk_index_to_local(global_chunk_index, documents, chunk_mapping)
    try:
        doc_text = documents[document_index]["doc_full_text"]
    except KeyError:
        doc_text = documents[document_index]["text"]
    doc_chunks = documents[document_index]["chunks"]
    chunk_boundaries = doc_chunks[local_chunk_index]
    chunk_text = doc_text[chunk_boundaries["start_char"] : chunk_boundaries["end_char"]]
    return chunk_text.strip()


def document_to_text(document_index: int, chunk_indices: list[int], documents: list[dict[str, Any]]) -> str:
    doc_text = documents[document_index]["doc_full_text"]
    doc_chunks = documents[document_index]["chunks"]
    doc_title = " ".join(documents[document_index]["doc_title"].strip().split()).strip()
    doc_annotation = " ".join(documents[document_index]["doc_annotation"].strip().split()).strip()
    if len(doc_title) > 0:
        new_document_description = f"**Название документа:** {doc_title}"
        if doc_title[-1] not in {".", ",", "?", "!", ":", ";"}:
            new_document_description += "."
    else:
        new_document_description = "Документ без названия."
    if len(doc_annotation) > 0:
        new_document_description += f"\n\n**Аннотация документа:** {doc_annotation}"
        if doc_annotation[-1] not in {".", ",", "?", "!", ":", ";"}:
            new_document_description += "."
    else:
        new_document_description += "\n\nДокумент без аннотации."

    local_indices_of_relevant_chunks = sorted(chunk_indices)
    num_relevant_chunks = len(local_indices_of_relevant_chunks)
    if num_relevant_chunks > 1:
        num_relevant_chunks_as_str = str(num_relevant_chunks)
        new_document_description += f"\n\n**{num_relevant_chunks_as_str} наиболее "
        if num_relevant_chunks_as_str[-1] == "1":
            new_document_description += "релеватный фрагмент документа:**"
        elif num_relevant_chunks_as_str[-1] in {"2", "3", "4"}:
            new_document_description += "релеватных фрагмента документа:**"
        else:
            new_document_description += "релеватных фрагментов документа:**"
        for idx, val in enumerate(local_indices_of_relevant_chunks):
            chunk_boundaries = doc_chunks[val]
            chunk_text = doc_text[chunk_boundaries["start_char"] : chunk_boundaries["end_char"]]
            new_document_description += f"\n\n<Фрагмент {idx + 1}>\n{chunk_text}\n</Фрагмент {idx + 1}>"
    else:
        chunk_boundaries = doc_chunks[local_indices_of_relevant_chunks[0]]
        chunk_text = doc_text[chunk_boundaries["start_char"] : chunk_boundaries["end_char"]]
        new_document_description += f"\n\n**Один самый релеватный фрагмент документа:**\n\n{chunk_text}"
    return new_document_description


def prepare_relevant_documents(
    indices_of_relevant_chunks: list[int],
    scores_of_relevant_chunks: list[float],
    documents: list[dict[str, Any]],
    chunk_mapping: dict[str, dict[str, int]],
    min_document_quality: float,
) -> dict[int, dict[str, Any]]:
    selected_documents: dict[int, dict[str, Any]] = {}
    for global_index, relevance in zip(indices_of_relevant_chunks, scores_of_relevant_chunks, strict=False):
        document_index, local_chunk_index = global_chunk_index_to_local(global_index, documents, chunk_mapping)
        if float(documents[document_index].get("quality_score", 1.0)) >= min_document_quality:
            if document_index in selected_documents:
                if relevance > selected_documents[document_index]["relevance"]:
                    selected_documents[document_index]["relevance"] = relevance
                selected_documents[document_index]["chunks"].append(local_chunk_index)
            else:
                selected_documents[document_index] = {"relevance": relevance, "chunks": [local_chunk_index]}
    return selected_documents


def prepare_context(
    indices_of_relevant_chunks: list[int],
    scores_of_relevant_chunks: list[float],
    documents: list[dict[str, Any]],
    chunk_mapping: dict[str, dict[str, int]],
    min_document_quality: float,
) -> tuple[list[str], list[str]]:
    selected_documents = prepare_relevant_documents(
        indices_of_relevant_chunks,
        scores_of_relevant_chunks,
        documents,
        chunk_mapping,
        min_document_quality,
    )
    if len(selected_documents) == 0:
        return [], []
    descriptions_of_selected_documents: list[str] = []
    descriptions_of_urls: list[str] = []
    ordered_indices_of_selected_documents = sorted(
        selected_documents.keys(),
        key=lambda idx: selected_documents[idx]["relevance"],
    )
    max_number_width = len(f"{len(ordered_indices_of_selected_documents)}. ")
    for counter, document_index in enumerate(ordered_indices_of_selected_documents):
        descriptions_of_selected_documents.append(
            document_to_text(
                document_index=document_index,
                chunk_indices=selected_documents[document_index]["chunks"],
                documents=documents,
            )
        )
        number = f"{counter + 1}. "
        while len(number) < max_number_width:
            number += " "
        doc_url = documents[document_index]["url"]
        doc_title = documents[document_index]["doc_title"]
        if len(doc_title) > 0:
            new_reference = number + doc_title + "\n"
            new_reference += "".join([" " for _ in range(max_number_width)])
            new_reference += doc_url
        else:
            new_reference = number + doc_url
        descriptions_of_urls.append(new_reference)
    return descriptions_of_selected_documents, descriptions_of_urls


def references_to_sources(references: list[str]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for reference in references:
        lines = [line.rstrip() for line in reference.splitlines() if line.strip()]
        if not lines:
            continue
        first_line = lines[0].strip()
        title = first_line.split(". ", 1)[1].strip() if ". " in first_line else first_line
        url = lines[-1].strip()
        sources.append({"document_title": title, "source_url": url})
    return sources
