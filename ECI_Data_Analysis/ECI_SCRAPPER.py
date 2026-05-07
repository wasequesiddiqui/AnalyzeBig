import time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager


BASE = "https://results.eci.gov.in/ResultAcGenMay2026/"


def setup_driver():
    options = Options()
    options.add_argument("--headless")  # run in background
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    return driver


def parse_constituency(driver, url):
    data = []

    driver.get(url)

    try:
        # wait until table loads
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
    except:
        return []

    # constituency name
    try:
        title = driver.find_element(By.TAG_NAME, "h2").text
    except:
        title = "Unknown"

    # table rows
    rows = driver.find_elements(By.CSS_SELECTOR, "table tr")

    headers = [th.text.strip() for th in rows[0].find_elements(By.TAG_NAME, "th")]

    for row in rows[1:]:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) != len(headers):
            continue

        values = [c.text.strip() for c in cols]
        record = dict(zip(headers, values))

        record["Constituency"] = title
        record["URL"] = url

        data.append(record)

    return data


def scrape():
    driver = setup_driver()
    all_data = []

    # iterate states
    for state in range(1, 40):
        state_code = f"S{state:02d}"
        print(f"\n🔎 Checking {state_code}")

        found = False

        # iterate constituencies
        for c in range(1, 300):
            url = f"{BASE}Constituencywise{state_code}{c}.htm"

            try:
                records = parse_constituency(driver, url)

                if records:
                    print(f"✔ {url}")
                    all_data.extend(records)
                    found = True

                time.sleep(0.5)

            except Exception as e:
                continue

        if not found:
            print(f"❌ No data in {state_code}")

    driver.quit()
    return pd.DataFrame(all_data)


if __name__ == "__main__":
    df = scrape()

    if df.empty:
        print("❌ No data scraped")
    else:
        df.to_csv("eci_results_selenium.csv", index=False)
        df.to_excel("eci_results_selenium.xlsx", index=False)
        print("✅ Data saved successfully")