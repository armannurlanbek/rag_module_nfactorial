from pathlib import Path
from collections import defaultdict
from docling.document_converter import DocumentConverter

PDF_PATH = Path(__file__).parent / "kaztelecom.pdf"
PAGE_RANGE = (2, 3)

def extract_pages(pdf_path: Path, page_range: tuple) -> dict:
    converter = DocumentConverter()
    result = converter.convert(pdf_path, page_range=page_range)

    pages_content = defaultdict(list)
    for item, _level in result.document.iterate_items():
        if item.prov and hasattr(item, "text"):
            page_no = item.prov[0].page_no
            pages_content[page_no].append(item.text)

    return dict(pages_content)

def validate(pages_content: dict, expected_pages: tuple) -> bool:
    start, end = expected_pages
    expected = set(range(start, end + 1))
    extracted = set(pages_content.keys())

    missing = expected - extracted
    if missing:
        print(f"[WARN] Missing pages: {missing}")

    for page_no in sorted(extracted):
        text_blocks = pages_content[page_no]
        print(f"[INFO] Page {page_no}: {len(text_blocks)} text blocks, "
              f"{sum(len(t) for t in text_blocks)} chars")

    return not missing

def main():
    print(f"Extracting pages {PAGE_RANGE[0]}-{PAGE_RANGE[1]} from {PDF_PATH.name} ...")
    pages_content = extract_pages(PDF_PATH, PAGE_RANGE)

    valid = validate(pages_content, PAGE_RANGE)
    print(f"\nValidation {'passed' if valid else 'failed'}")

    for page_no in sorted(pages_content):
        print(f"\n{'='*40} PAGE {page_no} {'='*40}")
        print("\n".join(pages_content[page_no]))

if __name__ == "__main__":
    main()
