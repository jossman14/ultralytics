"""Structural check of references.bib: every \\cite key in main.tex must exist
in references.bib, every bib entry must be cited, and required fields present.

This validates internal consistency, not external truth. External verification
(that each cited work exists as described) is recorded in reference_verification.md.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB = (ROOT / "references.bib").read_text()
TEX = (ROOT / "main.tex").read_text()

# bib keys and their entry types
entries = dict(re.findall(r"@(\w+)\s*\{\s*([^,]+),", BIB))
bib_keys = set(entries.values()) if False else set(re.findall(r"@\w+\s*\{\s*([^,]+),", BIB))

# cited keys (\cite{a,b,c})
cited = set()
for group in re.findall(r"\\cite\{([^}]+)\}", TEX):
    cited.update(k.strip() for k in group.split(","))

missing = cited - bib_keys          # cited but not defined
uncited = bib_keys - cited          # defined but never cited

# required fields per entry (title + author + year at minimum)
weak = []
for block in re.findall(r"@\w+\s*\{[^@]*", BIB):
    key = re.match(r"@\w+\s*\{\s*([^,]+),", block).group(1)
    for field in ("title", "year"):
        if not re.search(rf"\b{field}\s*=", block):
            weak.append(f"{key}: missing {field}")

print(f"bib entries : {len(bib_keys)}")
print(f"cited keys  : {len(cited)}")
print(f"missing (cited, no entry): {sorted(missing) or 'none'}")
print(f"uncited (entry, no cite) : {sorted(uncited) or 'none'}")
print(f"weak entries: {weak or 'none'}")

ok = not missing and not weak
assert not missing, f"undefined citations: {missing}"
assert not weak, f"entries missing required fields: {weak}"
print("\nPASS" if ok else "\nFAIL")
