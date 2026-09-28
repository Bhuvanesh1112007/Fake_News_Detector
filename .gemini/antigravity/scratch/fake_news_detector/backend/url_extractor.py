"""
Web Article Scraper & URL Feature Extractor (Phase 13)
Module Owner: Mohamed Irfaan M & Harish Hariharan MU

Extracts article headline, body paragraphs, author bylines, and domain metadata
from any public news URL for instant 79-dimensional multimodal verification.
"""
import re
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from typing import Optional, Dict, Any


class ArticleHTMLParser(HTMLParser):
    """HTML Parser that selectively extracts text content from article tags while ignoring script/nav/styles."""
    def __init__(self):
        super().__init__()
        self.title = ""
        self.meta_author = ""
        self.meta_og_title = ""
        self.paragraphs = []
        self.current_tag = ""
        self.in_title = False
        self.in_p = False
        self.in_h1 = False
        self.h1_texts = []
        self.current_p_text = []
        self.ignore_tags = {"script", "style", "nav", "footer", "header", "noscript", "aside", "form"}
        self.tag_stack = []

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag.lower()
        self.tag_stack.append(self.current_tag)
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}

        # Check meta tags
        if tag.lower() == "meta":
            prop = attrs_dict.get("property", "").lower()
            name = attrs_dict.get("name", "").lower()
            content = attrs_dict.get("content", "").strip()

            if prop in ("og:title", "twitter:title") and not self.meta_og_title:
                self.meta_og_title = content
            elif name in ("author", "byl", "dc.creator") and not self.meta_author:
                self.meta_author = content

        elif tag.lower() == "title":
            self.in_title = True
        elif tag.lower() == "h1":
            self.in_h1 = True
        elif tag.lower() == "p":
            if not any(t in self.ignore_tags for t in self.tag_stack):
                self.in_p = True
                self.current_p_text = []

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if self.tag_stack and self.tag_stack[-1] == tag_lower:
            self.tag_stack.pop()

        if tag_lower == "title":
            self.in_title = False
        elif tag_lower == "h1":
            self.in_h1 = False
        elif tag_lower == "p":
            self.in_p = False
            p_str = " ".join(self.current_p_text).strip()
            if len(p_str) > 25 and not any(p_str.startswith(x) for x in ["Cookie", "Privacy", "Copyright", "All rights", "Sign up"]):
                self.paragraphs.append(p_str)
            self.current_p_text = []

    def handle_data(self, data):
        cleaned = data.strip()
        if not cleaned:
            return

        if self.in_title:
            self.title += " " + cleaned
        elif self.in_h1:
            self.h1_texts.append(cleaned)
        elif self.in_p:
            self.current_p_text.append(cleaned)


def extract_article_from_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Scrapes and parses a web news article from a given URL:
    Returns dict: {'title': str, 'text': str, 'source': str, 'author': str, 'url': str}
    """
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            html_content = response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        raise ValueError(f"Failed to fetch content from URL '{url}': {str(e)}")

    parser = ArticleHTMLParser()
    try:
        parser.feed(html_content)
    except Exception:
        pass

    # Extract best Title
    extracted_title = ""
    if parser.meta_og_title:
        extracted_title = parser.meta_og_title
    elif parser.h1_texts:
        extracted_title = " ".join(parser.h1_texts)
    elif parser.title:
        extracted_title = parser.title

    extracted_title = re.sub(r'\s+', ' ', extracted_title).strip()
    # Strip site name suffix like " - BBC News" or " | Reuters"
    if " - " in extracted_title:
        extracted_title = extracted_title.split(" - ")[0].strip()
    elif " | " in extracted_title:
        extracted_title = extracted_title.split(" | ")[0].strip()

    # Extract Body Text
    if parser.paragraphs:
        body_text = "\n\n".join(parser.paragraphs)
    else:
        # Fallback: Strip HTML tags via regex
        clean_html = re.sub(r'<script.*?</script>', ' ', html_content, flags=re.DOTALL)
        clean_html = re.sub(r'<style.*?</style>', ' ', clean_html, flags=re.DOTALL)
        body_text = re.sub(r'<.*?>', ' ', clean_html)
        body_text = re.sub(r'\s+', ' ', body_text).strip()

    author = parser.meta_author or "Unknown"

    if not extracted_title:
        extracted_title = f"News Article from {domain}"

    return {
        "title": extracted_title,
        "text": body_text[:5000],  # Take first 5000 chars of body
        "source": domain,
        "author": author,
        "url": url
    }
