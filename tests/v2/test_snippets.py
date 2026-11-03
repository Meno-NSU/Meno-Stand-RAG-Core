from meno_rag.v2.snippets import best_window, query_terms, snippet

FILLER = "Список студентов группы, распределение по дисциплинам по выбору, факультет информационных технологий. "


def long_text(needle, before=25, after=25):
    return FILLER * before + needle + " " + FILLER * after


def test_name_after_2000_chars_is_in_the_excerpt():
    text = long_text("Группа 22215: Иванов А. А., Солихов Руслан Махмаднабиевич, Петров Б. Б.")
    assert text.index("Солихов") > 2000
    out = snippet(text, query_terms(["Что ты знаешь про Руслана Солихова?"]), 300)
    assert "Солихов Руслан Махмаднабиевич" in out
    assert out.startswith("… ") and out.endswith(" …")


def test_no_overlap_gives_none():
    terms = query_terms(["Кто такой Солихов?"])
    assert snippet("Расписание занятий публикуется на сайте университета.", terms, 300) is None
    assert snippet(long_text("ничего нужного"), terms, 300) is None
    assert snippet("что угодно", {}, 300) is None and best_window("", terms) is None


def test_excerpt_length_is_bounded():
    terms = query_terms(["Солихов Руслан"])
    for size in (100, 200, 320):
        out = snippet(long_text("Солихов Руслан Махмаднабиевич"), terms, size)
        assert out and len(out) <= size, size
    short = "Солихов Руслан учится на ФИТ."
    assert snippet(short, terms, 300) == short  # a short matching text is shown whole, without ellipses


def test_terms_stems_exact_tokens_and_stop_words():
    terms = query_terms(["Что ты знаешь про посвящение ФИТ НГУ и сайт nsuhw в 2026 году? Расскажи, пожалуйста"])
    assert {"x:фит", "x:нгу", "x:nsuhw", "x:2026"} <= set(terms)
    assert "s:посвящен" in terms
    assert not any(k.endswith(("знаеш", "расскаж", "пожалуйст")) for k in terms)  # conversational words
    assert "s:что" not in terms and "s:про" not in terms  # short words


def test_exact_and_inflected_matches():
    page = long_text("Домашние задания сдаются через https://nsuhw.ru/ до пятницы.")
    out = snippet(page, query_terms(["Куда сдавать домашку по дискретке, nsuhw?"]), 250)
    assert "nsuhw.ru" in out
    # «Солихова» (question) and «Солихов» (document) stem differently but still match
    assert "Солихов" in snippet(long_text("Солихов Р. М."), query_terms(["про Солихова"]), 200)


def test_idf_weights_prefer_the_rare_term():
    text = long_text("Иртегов читает лекции.", before=10, after=0) + "Иртегов " + FILLER * 3 + \
        "Студенты хвалят Иртегова и его лекции по операционным системам."
    rare = {"s:иртег": 5.0, "s:лекц": 0.5}
    out = snippet(text, rare, 200)
    assert "Иртегов" in out
    idf = query_terms(["Иртегов лекции"], idf=lambda stem: 5.0 if stem.startswith("иртег") else 0.5)
    assert max(idf, key=idf.get).startswith("s:иртег")
    assert query_terms(["Иртегов"], idf=lambda stem: 1 / 0) == {"s:иртег": 1.0}  # a broken table: plain counts
