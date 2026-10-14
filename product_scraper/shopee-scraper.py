from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime

import time
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument(r"--user-data-dir=C:\Scraper\Selenium")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")
options.add_argument("--disable-gpu")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

phone_brands = [
    'Xiaomi', 'Realme', 'Samsung', 'Oppo', 'Vivo', 'Infinix', 'Tecno', 'Asus']

TOTAL_PAGE_TO_SCRAPE = 10


for brand in phone_brands:
    
    data = []
    
    for page in range(TOTAL_PAGE_TO_SCRAPE):
        
        print(f"Scraping {brand}...")
        
        url = f"https://shopee.co.id/search?keyword={brand} second&page={page}"
        driver.get(url)
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        time.sleep(5)
        items = driver.find_elements(By.CSS_SELECTOR, "li.shopee-search-item-result__item")


        for item in items:
            try:
                title = item.find_element(By.CSS_SELECTOR, ".line-clamp-2").text
            except:
                title = None

            try:
                price = item.find_element(By.CSS_SELECTOR, ".text-base\\/5").text
            except:
                price = None

            try:
                location = item.find_element(By.CSS_SELECTOR, ".text-shopee-black54").text
            except:
                location = None

            try:
                rating = item.find_element(By.CSS_SELECTOR, ".text-shopee-black87").text
            except:
                rating = None

            try:
                sold = item.find_element(By.XPATH, ".//div[contains(text(),'Terjual')]").text
            except:
                sold = None

            data.append({
                "brand": brand,
                "title": title,
                "price": price,
                "location": location,
                "rating": rating,
                "sold": sold
            })
            
    df = pd.DataFrame(data)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"../datasets/marketplace_products/shopee/{brand.lower()}_shopee_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")

driver.quit()