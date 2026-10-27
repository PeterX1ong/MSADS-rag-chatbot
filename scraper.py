"""
Web scraper for the UChicago MS in Applied Data Science website.
ADSP 32027: Generative AI Principles -- Course Project 1 (RAG knowledge base)

Crawls the program's main page and its sublinks (breadth-first, same domain,
restricted to program-related paths), extracts the main content of every page
as heading-structured text, and saves:

    data/raw_pages/<slug>.txt   one file per page (Source/Title header + body)
    data/pages.jsonl            one JSON record per page (url, title, text, ...)

Usage:
    python scraper.py

Requires: requests, beautifulsoup4, lxml
"""

import hashlib
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Comment, NavigableString, Tag

# =============================================================================
# CONFIG
# =============================================================================

SEED_URL = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"
DOMAIN = "datascience.uchicago.edu"

MAX_DEPTH = 3       # seed page = depth 0
MAX_PAGES = 200     # hard cap on pages saved
DELAY_S = 1.0       # politeness delay between requests
TIMEOUT_S = 20
USER_AGENT = "Mozilla/5.0 (compatible; UChicago-ADSP-RAG-student-project/1.0)"

# Only follow links whose path starts with one of these (keeps the crawl on the program)
ALLOWED_PREFIXES = (
    "/education/masters-programs/",
    "/education/tuition-fees-aid/",
    "/how-to-apply/",
    "/capstone-projects/",
    "/explore-the-ms-ads-campus/",
    "/about-us/",
    "/people/",
)
# /news/ posts are only followed if their slug mentions the program
NEWS_PREFIX = "/news/"
NEWS_KEYWORDS = ("ms-ads", "applied-data-science", "masters")

SKIP_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
                   ".zip", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".mp4")

OUTPUT_DIR = Path(__file__).parent / "data"
PAGES_DIR = OUTPUT_DIR / "raw_pages"
JSONL_PATH = OUTPUT_DIR / "pages.jsonl"

# Elements inside <main> that are navigation / boilerplate, not content
NOISE_SELECTORS = [
    "script", "style", "noscript", "iframe", "svg", "form", "button", "nav",
    "aside.sidebar", ".mobile-sidebar-container", ".social-links",
    # "Related News, Insights, and Past Events" card grids (markup differs per page type)
    "section.section--ine", "section.section--related", ".content-section--related-content",
    ".divider", ".kicker",
]
# Where the program's sidebar menu lives (links are followed, text is discarded)
SIDEBAR_MENU_SELECTOR = "aside.sidebar .menu a, .mobile-sidebar .menu a"

BOILERPLATE_LINES = {"read more +", "read more", "show less -", "show less", "learn more",
                     "learn more +", "facet-arrow-down"}

# =============================================================================
# URL HANDLING
# =============================================================================

def normalize_url(href: str, base: str) -> str | None:
    """Resolve href against base and canonicalize it. Returns None if it should be skipped."""
    if not href:
        return None
    href = href.strip()
    if href.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None

    parsed = urlparse(urljoin(base, href))
    if parsed.scheme not in ("http", "https"):
        return None

    path = re.sub(r"(%20|\s)+$", "", parsed.path)  # e.g. ".../online-program/%20"
    path = re.sub(r"/{2,}", "/", path) or "/"
    if path.lower().endswith(SKIP_EXTENSIONS) or "/wp-content/" in path:
        return None
    if not path.endswith("/"):
        path += "/"

    # drop query string and fragment; force https
    return urlunparse(("https", parsed.netloc.lower(), path, "", "", ""))


def in_scope(url: str) -> bool:
    """True if url is on the DSI domain and under a program-related path."""
    parsed = urlparse(url)
    if parsed.netloc != DOMAIN:
        return False
    path = parsed.path
    if path.startswith(ALLOWED_PREFIXES):
        return True
    if path.startswith(NEWS_PREFIX):
        return any(k in path for k in NEWS_KEYWORDS)
    return False


def url_to_slug(url: str) -> str:
    """'https://.../ms-in-applied-data-science/faqs/' -> 'ms-in-applied-data-science__faqs'"""
    path = urlparse(url).path.strip("/")
    path = re.sub(r"^education/masters-programs/?", "", path) or "ms-in-applied-data-science"
    slug = path.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_\-]", "_", slug)

# =============================================================================
# HTML -> TEXT
# =============================================================================

def get_main_content(soup: BeautifulSoup) -> Tag | None:
    """Return the cleaned <main> element with navigation / boilerplate removed."""
    main = soup.select_one("main#main") or soup.find("main") or soup.body
    if main is None:
        return None
    for selector in NOISE_SELECTORS:
        for el in main.select(selector):
            el.decompose()
    # Course-progression schedules show a group label (Core / Elective / Optional ...) once,
    # before its courses; fold it into each course title so it stays attached after chunking
    for group in main.select(".course-group"):
        indicator = group.select_one(".course-group__indicator")
        if indicator is None:
            continue
        label = _clean(indicator.get_text(" "))
        indicator.decompose()
        for title in group.select("a.accordion-title"):
            title.append(f" [{label}]")
    for comment in main.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    return main


INLINE_TAGS = {"a", "span", "strong", "b", "em", "i", "u", "small", "sup", "sub",
               "code", "abbr", "mark", "label", "time", "br", "img"}
HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
    return re.sub(r"\s+([.,;:!?)])", r"\1", text)  # "applications ." -> "applications."


NESTING_CLASSES = {"accordion-item", "accordion__item", "quarter"}


def _heading_level(tag: Tag) -> int | None:
    """Heading level for real headings and for heading-like widgets (accordion/tab titles).

    Widget titles are nested by depth so the hierarchy survives, e.g.
    '### Core Courses' (outer tab) > '#### Machine Learning I' (inner item), or
    '### Sample Full-Time Schedule' > '#### Quarter 1' > '##### <course>'.
    """
    if tag.name in HEADING_TAGS:
        return HEADING_TAGS[tag.name]
    classes = tag.get("class", [])
    if "accordion-title" in classes or "quarter__title" in classes:
        depth = sum(1 for p in tag.parents if NESTING_CLASSES & set(p.get("class", [])))
        return min(6, 2 + depth)
    return None


def _render(el: Tag, out: list[str], bullet: bool = False) -> None:
    """Recursively turn el into lines: headings -> '#', list items -> '- ', blocks -> paragraphs."""
    buf: list[str] = []

    def flush():
        text = _clean(" ".join(buf))
        buf.clear()
        if text:
            out.append(f"- {text}" if bullet else text)

    for child in el.children:
        if isinstance(child, NavigableString):
            buf.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue

        level = _heading_level(child)
        if level is not None:
            flush()
            text = _clean(child.get_text(" "))
            if text:
                out.append(f"{'#' * level} {text}")
        elif child.name == "img":
            continue  # images are out of scope for the KB
        elif child.name in INLINE_TAGS and not child.find(_is_block):
            buf.append(child.get_text(" "))
        elif child.name == "table":
            flush()
            for row in child.find_all("tr"):
                cells = [_clean(c.get_text(" ")) for c in row.find_all(["th", "td"])]
                if any(cells):
                    out.append(" | ".join(cells))
        elif child.name == "li":
            flush()
            # accordion items are <li> too; only plain list items get a bullet
            is_widget = any(c.startswith("accordion") for c in child.get("class", []))
            _render(child, out, bullet=not is_widget)
        else:
            flush()
            _render(child, out, bullet=False)
    flush()


def _is_block(tag: Tag) -> bool:
    return isinstance(tag, Tag) and (tag.name not in INLINE_TAGS or _heading_level(tag) is not None)


def extract_text(main: Tag) -> str:
    """Convert the cleaned main content to markdown-ish text, dropping boilerplate lines."""
    lines: list[str] = []
    _render(main, lines)

    cleaned: list[str] = []
    for line in lines:
        bare = line.lstrip("#- ").strip().lower()
        if not bare or bare in BOILERPLATE_LINES:
            continue
        if cleaned and cleaned[-1] == line:  # consecutive duplicates
            continue
        if line.startswith("#") and cleaned:
            cleaned.append("")  # blank line before each heading for readability
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def get_title(soup: BeautifulSoup, main: Tag | None) -> str:
    h1 = main.find("h1") if main else None
    if h1 and _clean(h1.get_text(" ")):
        return _clean(h1.get_text(" "))
    if soup.title:
        return _clean(soup.title.get_text()).split(" - ")[0]
    return ""


def extract_links(anchors: list[Tag], base_url: str) -> list[str]:
    """Normalized, de-duplicated URLs from a list of <a> tags."""
    links = []
    for a in anchors:
        url = normalize_url(a.get("href", ""), base_url)
        if url and url not in links:
            links.append(url)
    return links

# =============================================================================
# CRAWLER
# =============================================================================

def load_robots() -> RobotFileParser:
    rp = RobotFileParser()
    rp.set_url(f"https://{DOMAIN}/robots.txt")
    try:
        rp.read()
    except Exception as e:  # if robots.txt is unreachable, proceed (allow_all)
        print(f"[warn] could not read robots.txt: {e}")
        rp.allow_all = True
    return rp


def scrape_page(session: requests.Session, url: str) -> tuple[str, BeautifulSoup] | None:
    """Fetch url; return (final_url, soup) or None if not an HTML page."""
    resp = session.get(url, timeout=TIMEOUT_S)
    resp.raise_for_status()
    if "text/html" not in resp.headers.get("Content-Type", ""):
        return None
    # pass raw bytes so BeautifulSoup honors the page's declared charset
    return resp.url, BeautifulSoup(resp.content, "lxml")


def crawl(seed: str = SEED_URL, max_depth: int = MAX_DEPTH, max_pages: int = MAX_PAGES) -> tuple[list[dict], list[tuple[str, str]]]:
    """Breadth-first crawl from seed. Returns (pages, failures)."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    robots = load_robots()

    queue = deque([(normalize_url(seed, seed), 0, None)])
    seen = {queue[0][0]}   # URLs already queued
    saved = set()          # final URLs (after redirects) already saved
    content_hashes = set()
    pages: list[dict] = []
    failures: list[tuple[str, str]] = []

    while queue and len(pages) < max_pages:
        url, depth, parent = queue.popleft()
        if url in saved:  # already saved via a redirect from another URL
            continue
        if not robots.can_fetch(USER_AGENT, url):
            failures.append((url, "disallowed by robots.txt"))
            continue

        try:
            result = scrape_page(session, url)
        except requests.RequestException as e:
            failures.append((url, str(e)))
            print(f"  [fail] {url} -> {e}")
            time.sleep(DELAY_S)
            continue
        time.sleep(DELAY_S)
        if result is None:
            failures.append((url, "not HTML"))
            continue

        final_url, soup = result
        final_url = normalize_url(final_url, final_url) or url
        if final_url in saved:
            failures.append((url, f"redirects to already-saved {final_url}"))
            continue
        seen.add(final_url)

        # sidebar links must be collected before get_main_content() strips the sidebar
        links = extract_links(soup.select(SIDEBAR_MENU_SELECTOR), final_url)
        main = get_main_content(soup)
        text = extract_text(main) if main is not None else ""
        title = get_title(soup, main)

        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        if not text or digest in content_hashes:
            failures.append((final_url, "empty or duplicate content"))
            continue
        content_hashes.add(digest)
        saved.add(final_url)

        pages.append({
            "url": final_url,
            "title": title,
            "depth": depth,
            "parent_url": parent,
            "n_chars": len(text),
            "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "text": text,
        })
        print(f"  [{len(pages):>3}] depth={depth} chars={len(text):>6}  {final_url}")

        if depth >= max_depth:
            continue
        if main is not None:
            links += extract_links(main.find_all("a", href=True), final_url)
        for link in links:
            if link not in seen and in_scope(link):
                seen.add(link)
                queue.append((link, depth + 1, final_url))

    return pages, failures

# =============================================================================
# OUTPUT
# =============================================================================

def save_pages(pages: list[dict]) -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    for old in PAGES_DIR.glob("*.txt"):  # fresh snapshot each run
        old.unlink()

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    for page in pages:
        header = f"Source: {page['url']}\nTitle: {page['title']}\n\n"
        (PAGES_DIR / f"{url_to_slug(page['url'])}.txt").write_text(header + page["text"], encoding="utf-8")


def print_summary(pages: list[dict], failures: list[tuple[str, str]]) -> None:
    print("\n" + "=" * 70)
    print(f"Scraped {len(pages)} pages | {sum(p['n_chars'] for p in pages):,} total chars")
    print(f"Saved to: {PAGES_DIR}  and  {JSONL_PATH}")
    short = [p for p in pages if p["n_chars"] < 300]
    if short:
        print(f"\n{len(short)} page(s) with < 300 chars (check content selector):")
        for p in short:
            print(f"  - {p['url']} ({p['n_chars']} chars)")
    if failures:
        print(f"\n{len(failures)} skipped/failed URL(s):")
        for url, reason in failures:
            print(f"  - {url}: {reason}")
    print("=" * 70)


def main():
    print(f"Crawling {SEED_URL}\n  max_depth={MAX_DEPTH}, max_pages={MAX_PAGES}, delay={DELAY_S}s\n")
    pages, failures = crawl()
    save_pages(pages)
    print_summary(pages, failures)


if __name__ == "__main__":
    main()
