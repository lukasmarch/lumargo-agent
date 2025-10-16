import csv
import requests

# Ścieżka do pliku CSV
csv_file = "/Users/lukaszmarchlewicz/Desktop/lumargo-agent/data/gallery.csv"


def check_image_urls(csv_file):
    with open(csv_file, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_url = row.get("image_url")
            id_url = row.get("id")
            if image_url:
                try:
                    response = requests.head(image_url, timeout=5)
                    if response.status_code == 200:
                        print(f"OK: {id_url},{image_url}")
                    else:
                        print(f"ERROR {response.status_code}:{id_url},{image_url}")
                except requests.RequestException as e:
                    print(f"FAILED:{id_url},{image_url} ({e})")


# Uruchomienie funkcji
if __name__ == "__main__":
    check_image_urls(csv_file)
