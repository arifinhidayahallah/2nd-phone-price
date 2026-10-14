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

phone_brands = ['Xiaomi', 'Realme', 'Samsung', 'Oppo', 'Vivo', 'Infinix', 'Tecno', 'Asus']
SCROLL_LIMIT = 10

for brand in phone_brands:
    
    total_load_more = 0

    data = []
    
    url = f"https://www.tokopedia.com/search?condition=2&st=product&q={brand}"
    driver.get(url)
    time.sleep(5)
    
    prev_count = 0
    while True:
        if(total_load_more == SCROLL_LIMIT):
            break
        
        seen_links = set()
        
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        total_load_more += 1
        time.sleep(5)
        items = driver.find_elements(By.XPATH, "//img[@alt='product-image']/ancestor::a")
        current_count = len(items)
        print(f"Loaded items: {current_count}")

        if prev_count == current_count:
            break
        
        prev_count = current_count
        
        buttons = driver.find_elements(By.XPATH, "//button[.//span[text()='Muat Lebih Banyak']]")
        if buttons:
            driver.execute_script("arguments[0].click();", buttons[0])
            time.sleep(5)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            print("Tombol 'Muat Lebih Banyak' ditemukan !")
            
        time.sleep(5)

        for item in items:
            try:
                link = item.get_attribute("href")

                if link in seen_links:
                    continue
                
                seen_links.add(link)

                title = item.find_element(
                    By.XPATH,
                    ".//div[.//span and not(.//img)]//span"
                ).text

                price = item.find_element(
                    By.XPATH,
                    ".//div[starts-with(text(),'Rp')]"
                ).text

                rating = item.find_element(
                    By.XPATH,
                    ".//img[@alt='rating']/ancestor::div/following-sibling::span"
                ).text

                sold = item.find_element(
                    By.XPATH,
                    ".//span[contains(text(),'terjual')]"
                ).text

                location = item.find_element(
                    By.XPATH,
                    ".//div[.//span[contains(@class,'flip')]]//span[last()]"
                ).text

                data.append({
                    # "link": link,  
                    "brand": brand,
                    "title": title,
                    "price": price,
                    "rating": rating,
                    "sold": sold,
                    "location": location,
                })
            except:
                continue
        
    df = pd.DataFrame(data)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"../datasets/marketplace_products/tokopedia/{brand.lower()}_tokopedia_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")

driver.quit()

