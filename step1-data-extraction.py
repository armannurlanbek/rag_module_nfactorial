import base64
import json
from pathlib import Path

import openai
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

load_dotenv()

IMAGE_DIR = Path(__file__).parent / "image"
OUTPUT_FILE = Path(__file__).parent / "step1-output.json"

# Map filenames to logical page numbers
PAGE_IMAGES = {
    2: IMAGE_DIR / "image.png",
    3: IMAGE_DIR / "image (1).png",
}

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


def extract_text_from_image(image_path: Path) -> str:
    converter = DocumentConverter()
    result = converter.convert(image_path)
    texts = []
    for item, _level in result.document.iterate_items():
        if hasattr(item, "text") and item.text:
            texts.append(item.text)
    return "\n".join(texts)


def judge_with_llm(image_path: Path, extracted_text: str) -> dict:
    client = openai.OpenAI()
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
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

    text = response.choices[0].message.content
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    return json.loads(text[start_idx:end_idx])


def main():
    test_cases = []
    validation_results = []

    for page_no, image_path in sorted(PAGE_IMAGES.items()):
        print(f"\n[INFO] Processing page {page_no} — {image_path.name} ...")

        extracted_text = extract_text_from_image(image_path)
        print(f"[INFO] Extracted {len(extracted_text)} chars")

        test_cases.append({
            "pdf_page_screenshot": str(image_path),
            "extracted_text": extracted_text,
            "extraction_method": "docling",
        })

        print(f"[INFO] Judging page {page_no} with Visual LLM ...")
        scores = judge_with_llm(image_path, extracted_text)
        scores["page"] = page_no
        validation_results.append(scores)
        print(json.dumps(scores, ensure_ascii=False, indent=2))

    output = {
        "test_cases": test_cases,
        "validation_results": validation_results,
    }

    OUTPUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n[INFO] Results saved to {OUTPUT_FILE}")

    print("\n=== VALIDATION SUMMARY ===")
    print(json.dumps(validation_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
