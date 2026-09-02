# pip install newspaper3k

import newspaper
import time
import csv
import pandas as pd
from os.path import exists
import nltk

nltk.download("punkt_tab")

INPUT_FILE = "./mbfc.csv"
OUTPUT_FILE = "./data/content_final/articles.csv"
ARTICLES_NUM = 200
DOMAIN_NUM = 8052
DOMAIN_START = 0
REQUEST_DELAY = 1.0

header = [
    "url",
    "title",
    "authors",
    "scrap_date",
    "publish_date",
    "text",
    "keywords",
    "domain",
]


def remove_empty_lines(text: str) -> str:
    """
    Remove empty lines (including lines with only whitespace) from a given text.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    cleaned_lines = [line for line in text.splitlines() if line.strip()]
    return " ".join(cleaned_lines)


def load_domains():
    df = pd.read_csv(INPUT_FILE)

    domains = (
        df["domain"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.rstrip("/")
        .drop_duplicates()
        .tolist()
    )

    print(f"News sources found: {len(domains)}")

    return domains


def load_articles(domain):
    paper = newspaper.build(domain, language="en", memoize_articles=False)
    print(f"Articles discovered by newspaper3k: {len(paper.articles)}")

    for article_index, content in enumerate(
        paper.articles[:ARTICLES_NUM], start=1
    ):
        content_path = f"{OUTPUT_FILE}"

        print(f"[{article_index}/{ARTICLES_NUM}] - " f"{content.url}")

        try:
            content.download()
            content.parse()

            url = content.url.strip() if content.url else ""
            title = content.title.strip() if content.title else ""
            authors = ", ".join(content.authors) if content.authors else ""

            scrap_date = time.strftime("%Y%m%d_%H%M%S")
            publish_date = (
                content.publish_date.strftime("%Y-%m-%d %H:%M:%S")
                if content.publish_date
                else ""
            )

            text = remove_empty_lines(content.text) if content.text else ""
            content.nlp()
            keywords = ", ".join(content.keywords) if content.keywords else ""

            try:
                file_exists = exists(content_path)
                if not file_exists:
                    with open(content_path, "w", encoding="utf-8") as file:
                        writer = csv.DictWriter(
                            file,
                            delimiter=",",
                            lineterminator="\n",
                            fieldnames=header,
                        )
                        writer.writeheader()
                        writer.writerow(
                            {
                                "url": url,
                                "title": title,
                                "authors": authors,
                                "scrap_date": scrap_date,
                                "publish_date": publish_date,
                                "text": text,
                                "keywords": keywords,
                                "domain": domain,
                            }
                        )
                else:
                    with open(content_path, "a", encoding="utf-8") as file:
                        writer = csv.DictWriter(
                            file,
                            delimiter=",",
                            lineterminator="\n",
                            fieldnames=header,
                        )
                        writer.writerow(
                            {
                                "url": url,
                                "title": title,
                                "authors": authors,
                                "scrap_date": scrap_date,
                                "publish_date": publish_date,
                                "text": text,
                                "keywords": keywords,
                                "domain": domain,
                            }
                        )
                print(f"Articles saved to {content_path}")

            except Exception as e:
                print("Error:", e)

        except Exception as e:
            print(f"  ERROR while processing article: " f"{e}")
            time.sleep(REQUEST_DELAY)


def load_articles_by_domain(domains):
    for source_index, domain in enumerate(
        domains[DOMAIN_START:DOMAIN_NUM], start=DOMAIN_START
    ):

        print()
        print("=" * 70)
        print(f"SOURCE {source_index}/{DOMAIN_NUM}")
        print(f"Domain URL: {domain}")
        print("=" * 70)

        load_articles(domain)


domains = load_domains()
load_articles_by_domain(domains)
