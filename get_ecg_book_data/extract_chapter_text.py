from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import html as html_lib
import json
import re


class SpringerChapterParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_body = False
        self.body_depth = 0
        self.skip_depth = 0
        self.sections: list[dict[str, object]] = []
        self.current_section: dict[str, object] | None = None
        self.capture_tag: str | None = None
        self.buffer: list[str] = []
        self.skip_classes = {
            "app-article-access",
            "app-explore-related-subjects",
            "c-article-access-provider",
            "c-article-recommendations",
            "c-article-share-box",
            "c-article-sidebar",
            "c-article-section__figure",
            "c-article-section__figure-caption",
            "c-article-section__figure-content",
            "c-article-section__figure-description",
            "c-article-section__figure-item",
            "c-article-table",
            "c-article-table__figcaption",
        }
        self.void_tags = {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
        self.heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
        self.paragraph_tags = {"p", "li"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            if self.in_body:
                self.skip_depth += 1
            return

        if not self.in_body and tag == "div":
            for key, value in attrs:
                if key == "data-article-body" and value == "true":
                    self.in_body = True
                    self.body_depth = 1
                    return

        if not self.in_body:
            return

        if self.skip_depth > 0:
            if tag not in self.void_tags:
                self.skip_depth += 1
            return

        for key, value in attrs:
            if key == "class" and value:
                classes = set(value.split())
                if classes & self.skip_classes:
                    if tag not in self.void_tags:
                        self.skip_depth = 1
                    return

        self.body_depth += 1

        if tag in self.heading_tags or tag in self.paragraph_tags:
            self.capture_tag = tag
            self.buffer = []

    def handle_endtag(self, tag: str) -> None:
        if not self.in_body:
            return

        if tag in {"script", "style"}:
            if self.skip_depth > 0:
                self.skip_depth -= 1
            return

        if self.skip_depth > 0:
            self.skip_depth -= 1
            return

        if self.capture_tag == tag:
            text = normalize_text("".join(self.buffer))
            self.capture_tag = None
            self.buffer = []
            if text:
                if tag in self.heading_tags:
                    self.current_section = {"title": text, "paragraphs": []}
                    self.sections.append(self.current_section)
                else:
                    if self.current_section is None:
                        self.current_section = {"title": "Untitled", "paragraphs": []}
                        self.sections.append(self.current_section)
                    paragraphs = self.current_section["paragraphs"]
                    if isinstance(paragraphs, list):
                        if tag == "li":
                            paragraphs.append(f"- {text}")
                        else:
                            paragraphs.append(text)

        self.body_depth -= 1
        if self.body_depth <= 0:
            self.in_body = False
            self.body_depth = 0

    def handle_data(self, data: str) -> None:
        if not self.in_body or self.skip_depth > 0:
            return
        if self.capture_tag:
            self.buffer.append(data)


def normalize_text(text: str) -> str:
    text = html_lib.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sections(html_path: Path) -> list[dict[str, object]]:
    parser = SpringerChapterParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    return parser.sections


def flatten_sections(sections: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for section in sections:
        title = section.get("title")
        if isinstance(title, str) and title:
            lines.append(title)
        paragraphs = section.get("paragraphs")
        if isinstance(paragraphs, list):
            for paragraph in paragraphs:
                if isinstance(paragraph, str) and paragraph:
                    lines.append(paragraph)
    return lines


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_html = base_dir / "index.html"
    output_txt = base_dir / "chapter_text.txt"
    output_json = base_dir / "chapter_text.json"

    sections = extract_sections(input_html)
    output_txt.write_text("\n".join(flatten_sections(sections)), encoding="utf-8")

    data = {
        "source": "https://link.springer.com/chapter/10.1007/978-3-030-40341-6_7",
        "sections": sections,
    }
    output_json.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Wrote {output_txt}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
