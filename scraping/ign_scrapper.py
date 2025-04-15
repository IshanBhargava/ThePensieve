import os
import requests
from bs4 import BeautifulSoup

def scrape_fandom_page(url: str, output_filename: str):
    """
    Scrapes visible text content and tables (e.g. quests) from a Fandom wiki page
    and saves it to data/hogwarts_legacy/{output_filename}.txt
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    content_div = soup.find('div', class_='mw-parser-output')
    cleaned_parts = []
    known_quest_titles = set()

    if content_div:
        # Extract tables and flatten them
        tables = content_div.find_all('table', class_='wikitable')
        for table in tables:
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            rows = table.find_all('tr')[1:]
            for row in rows:
                cols = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                if len(cols) == len(headers):
                    entry = "\n".join([f"{headers[i]}: {cols[i]}" for i in range(len(headers))])
                    cleaned_parts.append(entry)
                    # Track quest title
                    for h, c in zip(headers, cols):
                        if h.lower().startswith("quest"):
                            known_quest_titles.add(c.strip().lower())

        # Extract paragraphs and list items, skipping redundant quest names
        paragraphs = content_div.find_all(['p', 'li'])
        for tag in paragraphs:
            text = tag.get_text().strip()
            if not text:
                continue
            if text.lower() in known_quest_titles:
                continue
            cleaned_parts.append(text)

    cleaned_text = "\n\n".join(cleaned_parts)

    output_dir = os.path.join("data", "hogwarts_legacy")
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{output_filename}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"✅ Saved: {file_path}")


# Example usage
if __name__ == "__main__":
    url = "https://harrypotter.fandom.com/wiki/Hogwarts_Legacy"
    output_file = "hogwarts_legacy_overview"
    scrape_fandom_page(url, output_file)