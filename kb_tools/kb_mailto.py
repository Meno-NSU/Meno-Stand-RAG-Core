import re
EMAIL = re.compile("[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")

def decode_mailto(href):
    if not href.startswith("mailto:"):
        return None
    value = href.removeprefix("mailto:")
    if "@" not in value:
        if value.count("AT") != 1 or "DOT" not in value.split("AT", 1)[1]:
            return None
        value = value.replace("AT", "@").replace("DOT", ".")
    if not EMAIL.fullmatch(value):
        return None
    local, domain = value.split("@")
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return None
    if not all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label) for label in domain.split(".")):
        return None
    return value
