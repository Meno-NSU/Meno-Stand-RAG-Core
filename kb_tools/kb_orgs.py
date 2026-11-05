"""Site -> organisation labels shared by chunking, titles and CAG scripts."""

ORG = {
    "nsu.ru": "НГУ",
    "pure.nsu.ru": "НГУ, исследовательский портал Pure",
    "sesc.nsu.ru": "СУНЦ НГУ (ФМШ)",
    "t.me": "НГУ, официальный Telegram-канал",
    "vk.com": "НГУ, официальное сообщество ВКонтакте",
    "sbras.info": "газета СО РАН «Наука в Сибири»",
    "ru.wikipedia.org": "Википедия",
    "sbras.ru": "СО РАН",
    "nsc.ru": "СО РАН / ННЦ",
    "ipgg.sbras.ru": "ИНГГ СО РАН",
    "inp.nsk.su": "ИЯФ СО РАН",
    "catalysis.ru": "Институт катализа СО РАН",
    "math.nsc.ru": "Институт математики СО РАН",
    "ict.nsc.ru": "ФИЦ ИВТ",
    "icmmg.nsc.ru": "ИВМиМГ СО РАН",
    "itp.nsc.ru": "Институт теплофизики СО РАН",
    "hydro.nsc.ru": "Институт гидродинамики СО РАН",
    "isp.nsc.ru": "ИФП СО РАН",
    "iae.nsk.su": "ИАиЭ СО РАН",
    "laser.nsc.ru": "Институт лазерной физики СО РАН",
    "niic.nsc.ru": "ИНХ СО РАН",
    "nioch.nsc.ru": "НИОХ СО РАН",
    "kinetics.nsc.ru": "ИХКГ СО РАН",
    "solid.nsc.ru": "ИХТТМ СО РАН",
    "igm.nsc.ru": "ИГМ СО РАН",
    "archaeology.nsc.ru": "ИАЭТ СО РАН",
    "philology.nsc.ru": "Институт филологии СО РАН",
    "philosophy.nsc.ru": "Институт философии и права СО РАН",
    "history.nsc.ru": "Институт истории СО РАН",
    "ieie.su": "ИЭОПП СО РАН",
    "icgbio.ru": "ФИЦ ИЦиГ СО РАН",
    "bionet.nsc.ru": "ФИЦ ИЦиГ СО РАН",
    "niboch.nsc.ru": "ИХБФМ СО РАН",
    "eco.nsc.ru": "ИСиЭЖ СО РАН",
    "csbg-nsk.ru": "ЦСБС СО РАН",
    "csbg.nsc.ru": "ЦСБС СО РАН",
    "iis.nsk.su": "ИСИ СО РАН",
    "issa-siberia.ru": "Институт почвоведения и агрохимии СО РАН",
    "mcb.nsc.ru": "ИМКБ СО РАН",
    "spsl.nsc.ru": "ГПНТБ СО РАН",
    "vector.nsc.ru": "ГНЦ ВБ «Вектор»",
    "srf-skif.ru": "ЦКП «СКИФ»",
    "academpark.com": "Академпарк",
    "misd.ru": "Институт горного дела СО РАН",
    "museum.sbras.ru": "Музей СО РАН",
    "kolcovo.ru": "Наукоград Кольцово",
    "akademgorodok2.ru": "«Академгородок 2.0»",
    "centerfm.ru": "ФИЦ ФТМ",
    "iimed.ru": "НИИТПМ — филиал ИЦиГ СО РАН",
    "dusoran.ru": "Дом учёных СО РАН",
    "academcity.org": "academcity.org (Академгородок)",
    "fap.sbras.ru": "СО РАН (fap.sbras.ru)",
    "tdisie.nsc.ru": "ИЭОПП СО РАН (tdisie.nsc.ru)",
    "ancient.hydro.nsc.ru": "Институт гидродинамики СО РАН",
}


def org_label(site):
    h = (site or "").lower().removeprefix("www.")
    best = None
    for k in ORG:
        if (h == k or h.endswith("." + k)) and (best is None or len(k) > len(best)):
            best = k
    if best == "nsu.ru" and h != "nsu.ru":
        return f"НГУ ({h})"
    return f"{ORG[best]} ({h})" if best and best not in ("nsu.ru", "t.me", "vk.com") else (ORG[best] if best else h)
