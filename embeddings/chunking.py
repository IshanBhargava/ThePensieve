import fitz  # PyMuPDF
from pathlib import Path
from typing import List
import json

CHUNK_SIZE = 500  # target words per chunk
CHUNK_OVERLAP = 50  # overlapping words between chunks


def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    return text


def sliding_window_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap  # move forward with overlap
    return chunks


def chunk_all_books(data_folder: str, output_file: str, extension: str):
    root = Path(data_folder)
    all_chunks = []

    if extension.strip().lower() == 'pdf':
        for pdf in root.glob("*.pdf"):
            book_name = pdf.stem.replace("_", " ").title()
            print(f"📖 Processing: {book_name}")
            text = extract_text_from_pdf(str(pdf))
            chunks = sliding_window_chunks(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "source": book_name,
                    "chunk_id": f"{book_name}_chunk{i}",
                    "text": chunk
                })

    elif extension.strip().lower() == 'txt':
        for txt in root.glob("*.txt"):
            book_name = txt.stem.replace("_", " ").title()
            print(f"📖 Processing: {book_name}")
            with open(txt, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = sliding_window_chunks(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "source": book_name,
                    "chunk_id": f"{book_name}_chunk{i}",
                    "text": chunk
                })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(all_chunks)} chunks to {output_file}")


def run_chunking(data_to_be_chunked: dict) -> dict:
    chunk_files = dict()
    for key, value in data_to_be_chunked.items():
        folder, output_file, extension = value
        chunk_all_books(folder, output_file, extension)
        chunk_files[key] = output_file
    return chunk_files


if __name__ == "__main__":
    default_data_to_be_chunked = {
        'HP': ["data/HarryPotterBooks", "data/chunks/harry_potter_chunks.json", 'pdf'],
        'HL': ["data/hogwarts_legacy", "data/chunks/hogwarts_legacy_chunks.json", 'txt'],
        'FB': ["data/FantasticBeastsScripts", "data/chunks/fantastic_beasts_chunks.json", 'pdf']
    }
    run_chunking(default_data_to_be_chunked)
