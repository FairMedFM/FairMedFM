"""MkDocs hook: plain-text copies of the documentation for search engines and AI assistants.

- ``llms-full.txt``: every page as Markdown in navigation order, with API references expanded from the docstrings.
- ``<page>/index.md``: the Markdown of each page next to its HTML (the llms.txt convention).
- FAQPage structured data (JSON-LD) on the FAQ page, built from its questions and answers.
- ``config.extra.fairmedfm_version``: the package version, for the structured data in ``overrides/main.html``.
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
pages: dict[str, tuple[str, str, str]] = {}  # url -> (title, canonical url, markdown)
order: list[str] = []  # page urls in navigation order
_package = None


def on_config(config):
    pages.clear()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    config.extra["fairmedfm_version"] = project["version"]
    return config


def on_nav(nav, config, files):
    order[:] = [page.url for page in nav.pages]
    return nav


def on_page_markdown(markdown, page, config, files):
    pages[page.url] = (page.title or config.site_name, page.canonical_url, _expand_api(markdown))
    return markdown


def on_post_page(output, page, config):
    if page.file.src_uri != "faq.md":
        return output
    questions = []
    for question, answer in re.findall(r"^## (.+?)\n(.*?)(?=^## |\Z)", pages[page.url][2], flags=re.M | re.S):
        questions.append({"@type": "Question", "name": _plain(question),
                          "acceptedAnswer": {"@type": "Answer", "text": _plain(answer)}})
    data = {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": questions}
    script = f'<script type="application/ld+json">{json.dumps(data, ensure_ascii=False)}</script>\n'
    return output.replace("</head>", script + "</head>", 1)


def on_post_build(config):
    site = Path(config.site_dir)
    parts = [f"# {config.site_name} documentation\n\n> {config.site_description}\n\nSource: {config.site_url}\n"]
    for url in order + [url for url in pages if url not in order]:
        if url not in pages:
            continue
        title, canonical, markdown = pages[url]
        body = _strip_front_matter(markdown).strip()
        parts.append(f"<!-- {canonical} -->\n\n{body}\n")
        target = site / url / "index.md" if url else site / "index.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"<!-- {canonical} -->\n\n{body}\n")
    (site / "llms-full.txt").write_text("\n\n---\n\n".join(parts))


def _strip_front_matter(markdown: str) -> str:
    return re.sub(r"\A---\n.*?\n---\n", "", markdown, flags=re.S)


def _plain(markdown: str) -> str:
    text = re.sub(r"```.*?```", "", markdown, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def _expand_api(markdown: str) -> str:
    """Replace mkdocstrings ``::: path`` blocks with the signature and docstring, read statically with griffe."""
    def expand(match):
        path, options = match.group(1), match.group(2) or ""
        obj = _object(path)
        if obj is None:
            return match.group(0)
        text = f"### {obj.name}\n\n```python\n{_signature(obj)}\n```\n\n{obj.docstring.value if obj.docstring else ''}\n"
        members = re.search(r"members:\s*\[([^\]]*)\]", options)
        for name in (m.strip() for m in members.group(1).split(",")) if members else []:
            member = obj.members.get(name)
            if member is not None and member.docstring:
                text += f"\n#### {obj.name}.{name}\n\n```python\n{_signature(member)}\n```\n\n{member.docstring.value}\n"
        return text

    return re.sub(r"^::: (\S+)\n((?:[ \t]+.*\n)*)", expand, markdown, flags=re.M)


def _object(path: str):
    global _package
    import griffe

    if _package is None:
        _package = griffe.load("fairmedfm", search_paths=[str(ROOT / "src")])
    try:
        return _package[path.removeprefix("fairmedfm.")]
    except KeyError:
        return None


def _signature(obj) -> str:
    if obj.kind.value == "class":
        init = obj.members.get("__init__")
        parameters = list(init.parameters)[1:] if init is not None else []
        prefix = f"class {obj.name}"
    else:
        parameters = list(obj.parameters)
        prefix = obj.name
    parts, keyword_only = [], False
    for parameter in parameters:
        kind = parameter.kind.value
        if parameter.name == "self":
            continue
        if kind == "keyword-only" and not keyword_only:
            parts.append("*")
            keyword_only = True
        if kind == "variadic positional":
            parts.append(f"*{parameter.name}")
            keyword_only = True
        elif kind == "variadic keyword":
            parts.append(f"**{parameter.name}")
        else:
            parts.append(parameter.name + (f"={parameter.default}" if parameter.default is not None else ""))
    returns = f" -> {obj.returns}" if obj.kind.value == "function" and obj.returns else ""
    return f"{prefix}({', '.join(parts)}){returns}"
