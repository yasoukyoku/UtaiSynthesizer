# Wiktionary content in this directory

`wikt_cache.json` (2,102 entries, fetched by `wikt.py` from https://de.wiktionary.org) and
`enwikt_cache.json` (1,991 entries, fetched by `enwikt.py` from https://en.wiktionary.org) are
**verbatim caches of Wiktionary page wikitext** (definitions, inflection tables and pronunciation
templates, not only IPA). They are kept so the German glide rulers can be re-run offline and give
the same verdicts.

- **Source:** Wiktionary, the free dictionary — the entry for each key `<title>` is
  `https://de.wiktionary.org/wiki/<title>` (resp. `https://en.wiktionary.org/wiki/<title>`); its
  author list is that page's history (`?action=history`).
- **Licence:** Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0,
  https://creativecommons.org/licenses/by-sa/4.0/); Wiktionary text is also available under the GNU
  Free Documentation License. Attribution: **Wiktionary contributors**.
- **Changes:** none — each value is the wikitext exactly as returned by the MediaWiki API at fetch time.
- These two files, and only these, are under CC BY-SA 4.0. The rest of this repository keeps its own
  licence (see `LICENSE` and `NOTICE.md`); the rulers only *read* the cached text.
