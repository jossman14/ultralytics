"""Resolve DOIs/titles against Crossref and emit IEEE-formatted references.

Everything printed here comes from Crossref metadata, so no citation is invented.
Entries that fail to resolve are reported, not guessed.
"""
import json, sys, time, urllib.parse, urllib.request

UA = "ijaas-revision/1.0 (mailto:heheakun12@gmail.com)"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def by_doi(doi):
    return _get("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]


def by_title(title):
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 3})
    items = _get("https://api.crossref.org/works?" + q)["message"]["items"]
    return items[0] if items else None


def initials(given):
    return " ".join(f"{p[0]}." for p in given.replace("-", " ").split() if p)


def ieee(m):
    au = m.get("author", []) or []
    names = [f"{initials(a.get('given',''))} {a.get('family','')}".strip() for a in au if a.get("family")]
    if not names:
        who = ""
    elif len(names) > 6:
        who = names[0] + " et al., "
    elif len(names) == 1:
        who = names[0] + ", "
    else:
        who = ", ".join(names[:-1]) + ", and " + names[-1] + ", "
    title = (m.get("title") or [""])[0].strip().rstrip(".")
    ctype = m.get("type", "")
    venue = (m.get("container-title") or [""])
    venue = venue[0] if venue else ""
    year = ""
    for k in ("published-print", "published-online", "issued", "created"):
        if m.get(k, {}).get("date-parts", [[None]])[0][0]:
            year = str(m[k]["date-parts"][0][0]); break
    vol = m.get("volume", "")
    iss = m.get("issue", "")
    pages = m.get("page", "")
    art = m.get("article-number", "")
    doi = m.get("DOI", "")
    bits = [f'{who}"{title},"']
    if ctype == "proceedings-article":
        bits.append(f"in *{venue}*," if venue else "")
    else:
        bits.append(f"*{venue}*," if venue else "")
    if vol:
        bits.append(f"vol. {vol},")
    if iss:
        bits.append(f"no. {iss},")
    if pages:
        bits.append(f"pp. {pages}," if "-" in pages else f"p. {pages},")
    elif art:
        bits.append(f"art. no. {art},")
    if year:
        bits.append(f"{year},")
    bits.append(f"doi: {doi}.")
    return " ".join(b for b in bits if b)


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    out, bad = [], []
    for e in spec:
        try:
            m = by_doi(e["doi"]) if e.get("doi") else by_title(e["title"])
            if m is None:
                bad.append(e); continue
            out.append(dict(key=e["key"], doi=m.get("DOI"), type=m.get("type"),
                            title=(m.get("title") or [""])[0], ieee=ieee(m)))
        except Exception as ex:
            bad.append(dict(e, error=str(ex)))
        time.sleep(0.2)
    json.dump(out, open("/home/ftib/ultralytics/revisi/out/refs_resolved.json", "w"), indent=1)
    for o in out:
        print(f'[{o["key"]}] ({o["type"]}) {o["ieee"]}')
    print("\n--- UNRESOLVED ---")
    for b in bad:
        print(b)
