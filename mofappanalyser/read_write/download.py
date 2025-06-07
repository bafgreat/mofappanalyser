#!/usr/bin/env python3
import requests
import time
import argparse
import html


def clean_bibtex_value(value):
    if not value:
        return ""
    return html.unescape(value.replace('\n', ' ').replace('{', '').replace('}', '')).strip()


def generate_bibtex(entry):
    doi = entry.get("DOI", "no-doi")
    title = clean_bibtex_value(entry.get("title", [""])[0])
    authors = " and ".join([f"{a.get('family', '')}, {a.get('given', '')}" for a in entry.get("author", [])])
    journal = clean_bibtex_value(entry.get("container-title", [""])[0])
    year = entry.get("published-print", {}).get("date-parts", [[None]])[0][0] or entry.get("published-online", {}).get("date-parts", [[None]])[0][0]
    volume = entry.get("volume", "")
    number = entry.get("issue", "")
    pages = entry.get("page", "")
    url = entry.get("URL", "")
    abstract = clean_bibtex_value(entry.get("abstract", ""))

    # Use first author's last name and year as BibTeX ID fallback
    key = f"{entry['author'][0]['family']}_{year}" if entry.get("author") and year else doi.replace("/", "_")

    bibtex = f"@article{{{key},\n"
    bibtex += f"  title={{ {title} }},\n"
    if authors:
        bibtex += f"  author={{ {authors} }},\n"
    if journal:
        bibtex += f"  journal={{ {journal} }},\n"
    if year:
        bibtex += f"  year={{ {year} }},\n"
    if volume:
        bibtex += f"  volume={{ {volume} }},\n"
    if number:
        bibtex += f"  number={{ {number} }},\n"
    if pages:
        bibtex += f"  pages={{ {pages} }},\n"
    bibtex += f"  doi={{ {doi} }},\n"
    bibtex += f"  url={{ {url} }},\n"
    if abstract:
        bibtex += f"  abstract={{ {abstract} }},\n"
    bibtex += "}\n"

    return bibtex


def get_crossref_bibtex(query, start_year, end_year, rows=100, delay=1, output="mofs_articles.bib", email="your@email.com"):
    base_url = "https://api.crossref.org/works"
    headers = {"User-Agent": email}
    bibtex_entries = []

    for year in range(start_year, end_year + 1):
        cursor = "*"
        total_fetched = 0

        while True:
            params = {
                "query": query,
                "filter": f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31,type:journal-article",
                "rows": rows,
                "cursor": cursor,
                "mailto": email
            }

            print(f"Fetching year {year}, cursor {cursor}")
            r = requests.get(base_url, params=params, headers=headers)

            if r.status_code != 200:
                print(f"Request failed for year {year} with status code {r.status_code}")
                print("Response content:", r.text)
                break

            try:
                data = r.json()
            except ValueError:
                print(f"Invalid JSON for year {year}. Raw response:")
                print(r.text)
                break

            items = data.get("message", {}).get("items", [])
            if not items:
                break

            for item in items:
                try:
                    bibtex = generate_bibtex(item)
                    bibtex_entries.append(bibtex)
                except Exception as e:
                    print(f"Error generating BibTeX for entry: {e}")
                time.sleep(delay)

            cursor = data["message"].get("next-cursor")
            if not cursor or total_fetched >= 1000:
                break
            total_fetched += len(items)

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n\n".join(bibtex_entries))
    print(f"Saved {len(bibtex_entries)} entries to {output}")


def main():
    parser = argparse.ArgumentParser(description="Download MOF-related articles from CrossRef and save as a BibTeX file (with abstract).")
    parser.add_argument("--query", type=str, default="MOF OR metal-organic framework", help="Search query (use logical OR for multiple terms)")
    parser.add_argument("--start", type=int, default=2015, help="Start year (inclusive)")
    parser.add_argument("--end", type=int, default=2024, help="End year (inclusive)")
    parser.add_argument("--rows", type=int, default=100, help="Rows per request (max 1000)")
    parser.add_argument("--delay", type=int, default=1, help="Delay between requests (seconds)")
    parser.add_argument("--output", type=str, default="mofs_articles.bib", help="Output BibTeX filename")
    args = parser.parse_args()

    get_crossref_bibtex(
        query=args.query,
        start_year=args.start,
        end_year=args.end,
        rows=args.rows,
        delay=args.delay,
        output=args.output,
    )


if __name__ == "__main__":
    main()
