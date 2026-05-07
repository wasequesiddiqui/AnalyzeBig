import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager


# ---------- SETUP SELENIUM ----------
def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver


# ---------- FETCH PAGE ----------
def fetch_page(driver, url):
    driver.get(url)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
    except:
        print("❌ Table not loaded")
        return None

    return driver.page_source


# ---------- PARSE TABLE ----------
def parse_html(html, url):
    soup = BeautifulSoup(html, "html.parser")

    # Constituency name
    title = soup.find("h2")
    constituency = title.text.strip() if title else "Unknown"

    tables = soup.find_all("table")
    if not tables:
        return []

    # Pick table with most rows (main results table)
    table = max(tables, key=lambda t: len(t.find_all("tr")))

    rows = table.find_all("tr")

    # Headers (handle th/td)
    header_cells = rows[0].find_all(["th", "td"])
    headers = [c.text.strip() for c in header_cells]

    data = []

    for row in rows[1:]:
        cols = row.find_all("td")

        if len(cols) < 2:
            continue

        values = [c.text.strip() for c in cols]

        record = {}
        for i in range(min(len(headers), len(values))):
            record[headers[i]] = values[i]

        record["Constituency"] = constituency
        record["URL"] = url

        data.append(record)

    return data


# ---------- MAIN FUNCTION ----------
def scrape_eci(url):
    driver = setup_driver()

    print(f"🔎 Scraping: {url}")

    html = fetch_page(driver, url)
    driver.quit()

    if not html:
        print("❌ Failed to fetch page")
        return pd.DataFrame()

    data = parse_html(html, url)

    if not data:
        print("❌ No data extracted")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    return df


# ---------- RUN ----------
if __name__ == "__main__":
    url = "https://results.eci.gov.in/ResultAcGenMay2026/ConstituencywiseS2512.htm"

    df = scrape_eci(url)

    if df.empty:
        print("❌ No data found")
    else:
        df.to_csv("eci_output.csv", index=False)
        print("✅ Data saved to eci_output.csv")
        print(df.head())