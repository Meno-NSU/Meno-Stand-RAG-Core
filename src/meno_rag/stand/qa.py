from datetime import datetime
from typing import Optional

from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.fewshots import FewshotExample
from meno_rag.stand.prompts import QA_SYSTEM_PROMPT
from meno_rag.stand.rewriting import find_candidates_to_abbreviations

__all__ = [
    "QA_SYSTEM_PROMPT",
    "system_prompt_with_datetime",
    "calculate_number_of_documents_in_context",
    "prepare_prompt_for_question_answering",
]


def system_prompt_with_datetime(now: datetime) -> str:
    return QA_SYSTEM_PROMPT.format(current_datetime=now.isoformat())


def calculate_number_of_documents_in_context(context: str) -> int:
    if len(context.strip()) == 0:
        return 0
    document_prefix = "==========\nDOCUMENT "
    start_char_idx = context.rfind(document_prefix)
    if start_char_idx < 0:
        raise ValueError(f"The input context is wrong!\n\n{context}")
    end_char_idx = context[(start_char_idx + 1) :].find("\n==========")
    if end_char_idx < 0:
        raise ValueError(f"The found context is wrong!\n\n{context}")
    end_char_idx += start_char_idx + 1
    return int(context[(start_char_idx + len(document_prefix)) : end_char_idx].strip())


def prepare_prompt_for_question_answering(
    user_question: str,
    dialogue_history: str,
    context: str,
    abbr_dict: dict,
    stemmer: Optional[SnowballStemmer] = None,
    fewshots: Optional[list[FewshotExample]] = None,
) -> str:
    num_relevant_documents = calculate_number_of_documents_in_context(context)
    found_abbreviations = find_candidates_to_abbreviations(
        source_text=context + "\n\n" + dialogue_history + "\n\n" + user_question,
        all_abbreviations=abbr_dict,
        stemmer=stemmer,
    )
    input_prompt = ""
    if len(found_abbreviations) > 0:
        input_prompt += "==========\nABBREVIATION DICTIONARY\n==========\n\n"
        input_prompt += (
            "**Природа данных:** предположительные расшифровки, подобранные автоматически"
            " по совпадению буквенной формы. Записи могут быть лишними, ошибочными или "
            "неоднозначными (омонимия). Используйте словарь как подсказку, а не как "
            "достоверный источник.\n\n"
        )
        input_prompt += "**Записи:**\n" + found_abbreviations
        input_prompt += "\n\n"
    if len(context) > 0:
        input_prompt += context + "\n\n"
    if len(dialogue_history) > 0:
        input_prompt += "==========\nDIALOGUE HISTORY\n==========\n\n"
        input_prompt += dialogue_history
        input_prompt += "\n\n"
    if fewshots:
        input_prompt += "==========\nFEW-SHOT EXAMPLES\n==========\n\n"
        input_prompt += (
            "Ниже приведены примеры корректных ответов на релевантные вопросы. "
            "Используйте их как ориентир по стилю, структуре и уровню детализации.\n\n"
        )
        for i, fs in enumerate(fewshots, 1):
            input_prompt += f"--- ПРИМЕР {i} ---\n"
            input_prompt += f"Вопрос: {fs.question}\n"
            input_prompt += f"Ответ: {fs.answer}\n\n"
    input_prompt += "==========\nCURRENT QUESTION\n==========\n\n"
    input_prompt += user_question + "\n\n"
    input_prompt += "==========\nINSTRUCTION\n==========\n\n"
    input_prompt += "Ответьте, пожалуйста, на вопрос из раздела CURRENT QUESTION"
    if len(context) > 0:
        if num_relevant_documents > 1:
            input_prompt += f", опираясь только на факты из разделов DOCUMENT 1–{num_relevant_documents}."
        else:
            input_prompt += ", опираясь только на факты из разделов DOCUMENT 1."
    else:
        input_prompt += "."
    if len(dialogue_history) > 0:
        input_prompt += (
            "\n\nРаздел DIALOGUE HISTORY используйте исключительно для разрешения местоимений, "
            "уточнений и восстановления темы — не извлекайте из него фактические сведения."
        )
    if len(found_abbreviations) > 0:
        input_prompt += "\n\nРаздел ABBREVIATION DICTIONARY используйте по следующим правилам:\n"
        input_prompt += (
            "1. Применяйте расшифровку из словаря, только если сокращение действительно встречается в CURRENT QUESTION"
        )
        if (len(context) > 0) or (len(dialogue_history) > 0):
            if (len(context) > 0) and (len(dialogue_history) > 0):
                if num_relevant_documents > 1:
                    input_prompt += f", DIALOGUE HISTORY или DOCUMENT 1–{num_relevant_documents}"
                else:
                    input_prompt += ", DIALOGUE HISTORY или DOCUMENT 1"
            elif len(context) > 0:
                if num_relevant_documents > 1:
                    input_prompt += f" или DOCUMENT 1–{num_relevant_documents}"
                else:
                    input_prompt += " или DOCUMENT 1"
            else:
                input_prompt += " или DIALOGUE HISTORY"
        input_prompt += " и по контексту является аббревиатурой."
        input_prompt += (
            "2. Если одной аббревиатуре соответствует несколько расшифровок (омонимия), "
            "выберите ту, которая согласуется с темой диалога и содержанием документов. "
            "Если однозначно выбрать невозможно — коротко отметьте неоднозначность в ответе.\n"
        )
        input_prompt += (
            "3. Если в тексте встречается буквосочетание, совпадающее с ключом словаря, "
            "но по контексту аббревиатурой не является (например, это команда, междометие, "
            "имя собственное, часть слова), — игнорируйте запись словаря.\n"
        )
        input_prompt += (
            "4. Если подходящая расшифровка в словаре отсутствует, а в документах её тоже нет, — "
            "не придумывайте расшифровку; упомяните сокращение как есть.\n"
        )
        input_prompt += (
            "5. Словарь не является источником фактов: сам факт наличия записи «X — Y» не доказывает, "
            "что Y упомянут в документах или релевантен вопросу."
        )
    return input_prompt
