import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

BASE_URL = "http://ufcstats.com/statistics/events/completed"
DATE_CUTOFF = datetime(2025, 7, 12)  # July 12, 2025


def get_event_links():
    print("[DEBUG] Fetching completed events page...")
    resp = requests.get(BASE_URL)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    event_rows = soup.find_all("tr", class_="b-statistics__table-row")
    print(f"[DEBUG] Found {len(event_rows)} event rows.")
    for row in event_rows:
        # Find event link
        event_link = row.find("a", href=True)
        if not event_link:
            continue
        event_url = event_link["href"]
        if not event_url.startswith("http"):
            event_url = "http://ufcstats.com" + event_url
        # Find event date
        date_cell = row.find_all("td")[-1] if row.find_all("td") else None
        if not date_cell:
            continue
        date_str = date_cell.text.strip()
        try:
            event_date = datetime.strptime(date_str, "%B %d, %Y")
        except Exception:
            continue
        if event_date > DATE_CUTOFF:
            print(f"[DEBUG] Event: {event_url} ({event_date})")
            links.append((event_url, event_date))
        print(f"[DEBUG] Fetching event details: {event_url}")
    return links


def get_fight_links(event_url):
    print(f"[DEBUG] Fetching fight links from event: {event_url}")
    resp = requests.get(event_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    fight_links = soup.select("a.b-link.b-link_style_black")
    print(f"[DEBUG] Found {len(fight_links)} fight links.")
    for fight in fight_links:
        href = fight.get("href")
        if href and "/fight-details/" in href:
            links.append(href)
    return links


def scrape_fight(fight_url):
    resp = requests.get(fight_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    # Example: extract winner, fighters, division, date, etc.
    fight_data = {}
    fight_data["url"] = fight_url
    fight_data["date"] = soup.select_one("span.b-statistics__date").text.strip() if soup.select_one("span.b-statistics__date") else None
    fight_data["division"] = soup.select_one("i.b-statistics__division").text.strip() if soup.select_one("i.b-statistics__division") else None
    fight_data["winner"] = soup.select_one("div.b-fight-details__person.b-fight-details__person_style_left .b-fight-details__person-status")
    if fight_data["winner"]:
        fight_data["winner"] = fight_data["winner"].text.strip()
    else:
        fight_data["winner"] = None
    # Add more fields as needed
    return fight_data


def main():
    events = get_event_links()
    print(f"[DEBUG] Total events after cutoff: {len(events)}")
    for event_url, event_date in events:
        print(f"[DEBUG] Processing event: {event_url} ({event_date})")
        event_id = get_event_details(event_url)
        print(f"[DEBUG] Event ID: {event_id}")
        fight_links = get_fight_links(event_url)
        print(f"[DEBUG] Fight links: {fight_links}")
        for fight_url in fight_links:
            print(f"[DEBUG] Processing fight: {fight_url}")
            if not fight_url.startswith("http"):
                fight_url = "http://ufcstats.com" + fight_url
            scrape_fight(fight_url, event_id)
            time.sleep(1)
    print(f"[DEBUG] Writing {len(event_rows)} events to event_details.csv")
    print(f"[DEBUG] Writing {len(fight_rows)} fights to fight_details.csv")
    print(f"[DEBUG] Writing {len(fighter_rows)} fighters to fighter_details.csv")
    print(f"[DEBUG] Writing {len(ufc_rows)} rows to UFC.csv")
    pd.DataFrame(event_rows, columns=["event_id","event_name","event_date","event_location"]).to_csv(f"{EVAL_DIR}/event_details.csv", index=False)
    pd.DataFrame(fight_rows, columns=["fight_id","event_id","red_id","blue_id","winner_id","division","method","round","time"]).to_csv(f"{EVAL_DIR}/fight_details.csv", index=False)
    pd.DataFrame(fighter_rows, columns=["fighter_id","name","record","height","weight","reach","stance","birthdate"]).to_csv(f"{EVAL_DIR}/fighter_details.csv", index=False)
    pd.DataFrame(ufc_rows, columns=["event_id","fight_id","red_id","blue_id","winner_id","division","method","round","time","event_date","event_name","event_location"]).to_csv(f"{EVAL_DIR}/UFC.csv", index=False)
