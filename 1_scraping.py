import requests
from bs4 import BeautifulSoup
import json

def fetch_wikipedia_article_titles(start_url, limit=1000):
    visited = set()
    to_visit = [start_url]
    articles = []

    while to_visit and len(articles) < limit:
        url = to_visit.pop(0)

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the title and main content
            title = soup.find("h1", {"id": "firstHeading"}).text
            if title in visited:
                continue
            visited.add(title)
            print(f"Fetching ({len(articles) + 1}/{limit}): {title}")
            paragraphs = soup.find_all("p")
            content = " ".join([p.text for p in paragraphs])

            articles.append({"title": title, "content": content})

            # Extract links for crawling
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith("/wiki/") and ":" not in href and href.replace("/wiki/", "") not in ["Main_Page", "Wikipedia", "Free_content", "Encyclopedia"]:
                    full_url = f"https://en.wikipedia.org{href}"
                    to_visit.append(full_url)

        except Exception as e:
            print(f"Error fetching {url}: {e}")

    return articles

# Start scraping
start_url = "https://en.wikipedia.org/wiki/Special:Random"
articles = fetch_wikipedia_article_titles(start_url)

# Save data to a JSON file
with open("wikipedia_articles.json", "w") as file:
    json.dump(articles, file)
