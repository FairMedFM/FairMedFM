"""Combine the project page with the built documentation into the GitHub Pages site.

Usage: python homepage/assemble_site.py SITE_DIR, after mkdocs has built the documentation into SITE_DIR/docs.
Needs the full git history (for the sitemap dates). Prints the URLs changed since yesterday, for IndexNow.
"""
import datetime
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SITE = "https://fairmedfm.github.io/FairMedFM/"
DOCS = SITE + "docs/"
HOMEPAGE = Path(__file__).resolve().parent
REPO = HOMEPAGE.parent
NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


def last_commit_date(*paths):
    """Date of the last commit touching any of the paths, as YYYY-MM-DD (None if unknown)."""
    result = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%cs", "--", *paths],
                            capture_output=True, text=True)
    return result.stdout.strip() or None


def doc_sources(url):
    """Files that a documentation page is built from."""
    slug = url[len(DOCS):].strip("/")
    page = f"docs/{slug or 'index'}.md"
    return [page, "src/fairmedfm"] if slug == "api" else [page]


def write_sitemap(entries, path):
    urlset = ET.Element(f"{{{NS}}}urlset")
    for loc, lastmod in entries:
        url = ET.SubElement(urlset, f"{{{NS}}}url")
        ET.SubElement(url, f"{{{NS}}}loc").text = loc
        if lastmod:
            ET.SubElement(url, f"{{{NS}}}lastmod").text = lastmod
    ET.ElementTree(urlset).write(path, encoding="utf-8", xml_declaration=True)


def main(output):
    output = Path(output)
    shutil.copy2(HOMEPAGE / "index.html", output / "index.html")
    shutil.copytree(HOMEPAGE / "static", output / "static")
    # Search engine ownership files and the IndexNow key, served from the site root.
    for path in (HOMEPAGE / "verification").glob("*"):
        shutil.copy2(path, output / path.name)
    (output / ".nojekyll").touch()

    # Sitemaps dated by the last change to each page's sources rather than the build date.
    ET.register_namespace("", NS)
    docs = [url.find(f"{{{NS}}}loc").text for url in ET.parse(output / "docs" / "sitemap.xml").getroot()]
    docs = [(loc, last_commit_date(*doc_sources(loc))) for loc in docs]
    entries = [(SITE, last_commit_date("homepage/index.html", "homepage/static"))] + docs
    write_sitemap(entries, output / "sitemap.xml")
    write_sitemap(docs, output / "docs" / "sitemap.xml")  # a sitemap may only list URLs below its own folder
    (output / "docs" / "sitemap.xml.gz").unlink(missing_ok=True)

    # robots.txt is only read at the host root (fairmedfm.github.io/robots.txt), so this copy is informational;
    # submit the sitemap in Google Search Console and Bing Webmaster Tools.
    (output / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {SITE}sitemap.xml\n")
    for name in ("llms.txt", "llms-full.txt"):
        shutil.copy2(output / "docs" / name, output / name)

    since = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    for loc, lastmod in entries:
        if lastmod and lastmod >= since:
            print(loc)


if __name__ == "__main__":
    main(sys.argv[1])
