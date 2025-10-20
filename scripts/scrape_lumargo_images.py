import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

# Konfiguracja
BASE_URL = "https://www.lumargo.pl"
OUTPUT_CSV = "../data/gallery_category.csv"

# Lista kategorii do przeskanowania
CATEGORIES = [
    "blaty",
    "schody",
    "nagrobki",
    "posadzki",
    "elewacje",
    "kominki",
    "lazienki",
]


def get_image_urls_from_page(url, category):
    """Pobiera URLs zdjęć z konkretnej strony"""
    images = []

    try:
        print(f"Pobieranie: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Szukaj wszystkich tagów <img>
        img_tags = soup.find_all("img")

        for img in img_tags:
            # Pobierz src lub data-src (dla lazy loading)
            img_url = img.get("src") or img.get("data-src") or img.get("data-lazy-src")

            if img_url:
                # Pomiń małe obrazki (ikony, logo itp.)
                if any(
                    skip in img_url.lower()
                    for skip in ["logo", "icon", "avatar", "sprite"]
                ):
                    continue

                # Konwertuj na pełny URL jeśli jest względny
                full_url = urljoin(BASE_URL, img_url)

                # Sprawdź czy to obrazek z wp-content (główne zdjęcia)
                if "wp-content/uploads" in full_url:
                    images.append({"url": full_url, "category": category})

        # Szukaj również w galeriach (często używane w WordPress)
        gallery_links = soup.find_all(
            "a",
            class_=lambda x: x and ("gallery" in x.lower() or "lightbox" in x.lower()),
        )
        for link in gallery_links:
            href = link.get("href")
            if href and any(
                ext in href.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]
            ):
                full_url = urljoin(BASE_URL, href)
                if "wp-content/uploads" in full_url:
                    images.append({"url": full_url, "category": category})

        print(f"Znaleziono {len(images)} zdjęć w kategorii {category}")

    except requests.RequestException as e:
        print(f"Błąd podczas pobierania {url}: {e}")

    return images


def scrape_all_categories():
    """Przeskanuj wszystkie kategorie"""
    all_images = []

    for category in CATEGORIES:
        url = f"{BASE_URL}/{category}"
        images = get_image_urls_from_page(url, category)
        all_images.extend(images)

        # Poczekaj chwilę między requestami żeby nie obciążać serwera
        time.sleep(1)

    return all_images


def save_to_csv(images, output_file):
    """Zapisz dane do pliku CSV"""
    if not images:
        print("Brak zdjęć do zapisania")
        return

    # Usuń duplikaty na podstawie URL
    unique_images = []
    seen_urls = set()

    for img in images:
        if img["url"] not in seen_urls:
            seen_urls.add(img["url"])
            unique_images.append(img)

    print(f"\nZapisywanie {len(unique_images)} unikalnych zdjęć do {output_file}")

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Nagłówki
        writer.writerow(["id", "image_url", "category"])

        # Dane
        for idx, img in enumerate(unique_images, start=1):
            img_id = f"IMG{idx:04d}"  # IMG0001, IMG0002, etc.
            writer.writerow([img_id, img["url"], img["category"]])

    print(f"Zapisano pomyślnie!")


def main():
    print("=== Scraper zdjęć z lumargo.pl ===\n")
    print(f"Kategorie do przeskanowania: {', '.join(CATEGORIES)}\n")

    # Pobierz zdjęcia
    images = scrape_all_categories()

    # Zapisz do CSV
    save_to_csv(images, OUTPUT_CSV)

    print("\n=== Zakończono ===")


if __name__ == "__main__":
    main()
