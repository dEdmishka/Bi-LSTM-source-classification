import pandas as pd
import glob
import os

FOLDER_PATH = "./data"
INPUT_FILE = "./mbfc.csv"
ARTICLES_FILE = "./articles.csv"
ARTICLES_DOMAINS_PRESENT_FILE = "articles_with_domains_present.csv"

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


def find_csv_files(base_path):
    """
    Find all CSV files in the given directory and its subdirectories.

    :param base_path: Path to the directory to search.
    :return: List of absolute file paths to CSV files.
    """
    if not os.path.isdir(base_path):
        raise NotADirectoryError(
            f"Provided path is not a directory: {base_path}"
        )

    csv_files = glob.glob(
        os.path.join(base_path, "**", "*.csv"), recursive=True
    )

    csv_files = [os.path.abspath(f) for f in csv_files if os.path.isfile(f)]

    return csv_files


def glob_articles():
    try:
        folder_path = FOLDER_PATH
        files = find_csv_files(folder_path)

        if files:
            print(f"Found {len(files)} CSV file(s):")
            for f in files:
                print(f)

            dfs = []
            for file in files:
                try:
                    df = pd.read_csv(file)
                    dfs.append(df)
                    print(f"Loaded: {file} ({len(df)} rows)")
                except Exception as e:
                    print(f"Skipping {file} due to error: {e}")

            combined_df = pd.concat(dfs, ignore_index=True)

            print(f"Starting df Len is {len(combined_df)}")

            combined_df = combined_df.drop_duplicates()

            print(f"Dropped duplicates df Len is {len(combined_df)}")

            combined_df = combined_df[
                combined_df["text"].str.len() >= 100
            ].reset_index(drop=True)

            print(f"Dropped short text df Len is {len(combined_df)}")

            combined_df.to_csv("articles_full.csv", index=False)

            print(f"CSV files merged successfully! Len is {len(combined_df)}")
        else:
            print("No CSV files found.")
    except Exception as e:
        print(f"Error: {e}")


def merge_origins_and_articles():
    try:
        mbfc = pd.read_csv(INPUT_FILE)
        articles = pd.read_csv(ARTICLES_FILE)

        def normalize_domain(domain):
            if pd.isna(domain):
                return domain

            domain = str(domain).lower().strip().rstrip("/")

            return domain

        mbfc = mbfc.drop_duplicates(subset=["domain"])

        articles["domain"] = articles["domain"].apply(normalize_domain)
        mbfc["domain"] = mbfc["domain"].apply(normalize_domain)

        articles_enriched = articles.merge(
            mbfc[
                ["domain", "bias_rating", "factuality_rating", "credibility"]
            ],
            on="domain",
            how="left",
            validate="many_to_one",
        )

        print(f"Starting df Len is {len(articles_enriched)}")

        articles_enriched = articles_enriched.fillna("")

        articles_enriched = articles_enriched.drop_duplicates()

        print(f"Dropped duplicates df Len is {len(articles_enriched)}")

        articles_enriched = articles_enriched[
            articles_enriched["text"].str.len() >= 100
        ].reset_index(drop=True)

        print(f"Dropped short text df Len is {len(articles_enriched)}")

        articles_enriched = articles_enriched[
            articles_enriched["factuality_rating"] != "NOT RATED"
        ].reset_index(drop=True)

        print(
            f"Dropped NOT RATED factuality df Len is {len(articles_enriched)}"
        )

        articles_enriched = articles_enriched[
            articles_enriched["credibility"] != "NOT RATED"
        ].reset_index(drop=True)

        print(
            f"Dropped NOT RATED credibility df Len is {len(articles_enriched)}"
        )

        articles_enriched = articles_enriched[
            articles_enriched["factuality_rating"] != "VERY LOW"
        ].reset_index(drop=True)

        print(
            f"Dropped VERY LOW factuality_rating df Len is {len(articles_enriched)}"
        )

        articles_enriched = articles_enriched[
            articles_enriched["factuality_rating"] != "MOSTLY FACTUAL"
        ].reset_index(drop=True)

        print(
            f"Dropped MOSTLY FACTUAL factuality_rating df Len is {len(articles_enriched)}"
        )

        articles_enriched = articles_enriched[
            articles_enriched["factuality_rating"] != "VERY HIGH"
        ].reset_index(drop=True)

        print(
            f"Dropped VERY HIGH factuality_rating df Len is {len(articles_enriched)}"
        )

        print(
            f"LOW factuality_rating df Len is {len(articles_enriched[articles_enriched['factuality_rating'] == 'LOW'])}"
        )

        print(
            f"MIXED factuality_rating df Len is {len(articles_enriched[articles_enriched['factuality_rating'] == 'MIXED'])}"
        )

        print(
            f"HIGH factuality_rating df Len is {len(articles_enriched[articles_enriched['factuality_rating'] == 'HIGH'])}"
        )

        articles_enriched.to_csv(
            "articles_with_domains_present.csv", index=False
        )

        print("Готово.")
        print(f"Кількість статей: {len(articles_enriched)}")
        print(
            f"Статей без MBFC-оцінок: "
            f"{articles_enriched['factuality_rating'].isna().sum()}"
        )

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        # glob_articles()
        # merge_origins_and_articles()

        df = pd.read_csv(ARTICLES_DOMAINS_PRESENT_FILE)

        n_domains = df["domain"].nunique()

        print(f"unique_domains={n_domains}")

        df["bias_rating"] = df["bias_rating"].astype("category")

        print(
            "Categories bias_rating:",
            df["bias_rating"].cat.categories.tolist(),
        )

        df["credibility"] = df["credibility"].astype("category")

        print(
            "Categories credibility:",
            df["credibility"].cat.categories.tolist(),
        )

        df["factuality_rating"] = df["factuality_rating"].astype("category")

        print(
            "Categories factuality_rating:",
            df["factuality_rating"].cat.categories.tolist(),
        )

        print(
            "Categories factuality_rating is LOW:",
            len(df[df["factuality_rating"] == "LOW"]),
        )

        print(
            "Categories factuality_rating is MIXED:",
            len(df[df["factuality_rating"] == "MIXED"]),
        )

        print(
            "Categories factuality_rating is HIGH:",
            len(df[df["factuality_rating"] == "HIGH"]),
        )

    except Exception as e:
        print(f"Error: {e}")
