import base64
import json
from collections import defaultdict
from pathlib import Path

import anthropic
import pypdfium2 as pdfium
from docling.document_converter import DocumentConverter

PDF_PATH = Path(__file__).parent / "kaztelecom.pdf"
PAGE_RANGE = (2, 3)
SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"

JUDGE_PROMPT = """
Тебе даны:
1. Скриншот оригинальной страницы PDF
2. Извлечённый текст/Markdown из этой страницы

Оцени от 1 до 5 качество извлечения по следующим критериям:

**Структура (1-5):**
- Сохранена ли структура заголовков (H1, H2, H3)?
- Правильно ли распознаны списки (нумерованные, маркированные)?

**Таблицы (1-5):**
- Сохранена ли структура таблиц?
- Читаемы ли данные в ячейках?
- Правильно ли выровнены столбцы и строки?

**Форматирование (1-5):**
- Сохранены ли жирные, курсивные элементы?
- Правильно ли обработаны формулы/спецсимволы?

**Полнота (1-5):**
- Весь ли текст извлечён?
- Нет ли пропущенных блоков?

**Итоговая оценка:** (среднее значение)

Ответ дай строго в формате JSON (без лишнего текста):
{
  "structure_score": X,
  "tables_score": X,
  "formatting_score": X,
  "completeness_score": X,
  "overall_score": X,
  "comments": "..."
}
"""


def extract_pages(pdf_path: Path, page_range: tuple) -> dict:
    converter = DocumentConverter()
    result = converter.convert(pdf_path, page_range=page_range)

    pages_content = defaultdict(list)
    for item, _level in result.document.iterate_items():
        if item.prov and hasattr(item, "text"):
            page_no = item.prov[0].page_no
            pages_content[page_no].append(item.text)

    return dict(pages_content)


def render_screenshots(pdf_path: Path, page_range: tuple, out_dir: Path) -> dict:
    out_dir.mkdir(exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    screenshots = {}
    start, end = page_range
    for page_no in range(start, end + 1):
        page = pdf[page_no - 1]  # pypdfium2 is 0-indexed
        bitmap = page.render(scale=2)
        image = bitmap.to_pil()
        path = out_dir / f"page{page_no}.png"
        image.save(path)
        screenshots[page_no] = path
        print(f"[INFO] Screenshot saved: {path}")
    return screenshots


def judge_with_llm(page_no: int, screenshot_path: Path, extracted_text: str) -> dict:
    client = anthropic.Anthropic()
    with open(screenshot_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"{JUDGE_PROMPT}\n\nИзвлечённый текст:\n{extracted_text}",
                    },
                ],
            }
        ],
    )

    text = response.content[0].text
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    return json.loads(text[start_idx:end_idx])


def main():
    print(f"Extracting pages {PAGE_RANGE[0]}-{PAGE_RANGE[1]} from {PDF_PATH.name} ...")
    pages_content = extract_pages(PDF_PATH, PAGE_RANGE)

    print("\nRendering page screenshots ...")
    screenshots = render_screenshots(PDF_PATH, PAGE_RANGE, SCREENSHOTS_DIR)

    test_cases = []
    validation_results = []

    for page_no in sorted(pages_content):
        extracted_text = "\n".join(pages_content[page_no])

        test_cases.append({
            "pdf_page_screenshot": str(screenshots[page_no]),
            "extracted_text": extracted_text,
            "extraction_method": "docling",
        })

        print(f"\n[INFO] Judging page {page_no} with Visual LLM ...")
        scores = judge_with_llm(page_no, screenshots[page_no], extracted_text)
        scores["page"] = page_no
        validation_results.append(scores)
        print(json.dumps(scores, ensure_ascii=False, indent=2))

    print("\n=== TEST CASES ===")
    print(json.dumps(test_cases, ensure_ascii=False, indent=2))

    print("\n=== VALIDATION SUMMARY ===")
    print(json.dumps(validation_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
