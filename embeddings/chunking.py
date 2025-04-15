import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List
import json

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts full text from a PDF file."""
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    return text


def split_into_chunks(text: str, max_tokens: int = 500) -> List[str]:
    """Splits text into chunks of roughly max_tokens words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
    return chunks


def chunk_all_books(data_folder: str, output_file: str, extension: str):
    """Processes all PDFs in a folder and outputs chunked text with metadata."""
    root = Path(data_folder)
    all_chunks = []

    if extension.strip().lower() == 'pdf':
        for pdf in root.glob("*.pdf"):
            book_name = pdf.stem.replace("_", " ").title()
            print(f"📖 Processing: {book_name}")

            text = extract_text_from_pdf(str(pdf))
            chunks = split_into_chunks(text)

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

            chunks = split_into_chunks(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "source": book_name,
                    "chunk_id": f"{book_name}_chunk{i}",
                    "text": chunk
                })

    # Save as JSON for later embedding
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(all_chunks)} chunks to {output_file}")



if __name__ == "__main__":
    data_to_be_chunked = {
        'HP': ["data/HarryPotterBooks", "data/chunks/harry_potter_chunks.json", 'pdf'],
        'HL': ["data/hogwarts_legacy", "data/chunks/hogwarts_legacy_chunks.json", 'txt'],
        'FB': ["data/FantasticBeastsScripts", "data/chunks/fantastic_beasts_chunks.json", 'pdf']
    }

    for key, value in data_to_be_chunked.items():
        folder, output_file, extension = value
        chunk_all_books(folder, output_file, extension)
    
    # chunk_all_books("data/HarryPotterBooks", "data/chunks/harry_potter_chunks.json", 'pdf')
    # chunk_all_books("data/hogwarts_legacy", "data/chunks/hogwarts_legacy_chunks.json", 'txt')
    # chunk_all_books("data/FantasticBeastsScripts", "data/chunks/fantastic_beasts_chunks.json", 'pdf')