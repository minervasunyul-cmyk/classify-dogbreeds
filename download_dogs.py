import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        WEBDRIVER_MANAGER_AVAILABLE = True
    except ImportError:
        WEBDRIVER_MANAGER_AVAILABLE = False
except ImportError:
    SELENIUM_AVAILABLE = False
    WEBDRIVER_MANAGER_AVAILABLE = False
import json
import re

########################################################
prefix = "pomeranian5"
URL = "https://www.google.com/search?q=pomeranian+pic&sca_esv=6094bdfe284e5b6e&udm=2&biw=1920&bih=869&ei=Ut2ZafyGEvGoptQP-6Wy0QU&ved=0ahUKEwj89-PKgOuSAxVxlIkEHfuSLFoQ4dUDCBQ&uact=5&oq=pomeranian+pic&gs_lp=Egtnd3Mtd2l6LWltZyIOcG9tZXJhbmlhbiBwaWMyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABDIEEAAYHjIEEAAYHjIEEAAYHjIGEAAYChgeMgQQABgeSOsIUKcFWPEHcAJ4AJABAJgBLaABhwGqAQEzuAEDyAEA-AEBmAIFoAKcAcICChAAGIAEGEMYigXCAgYQABgHGB7CAggQABiABBixA8ICBxAAGIAEGAqYAwCIBgGSBwE1oAfoD7IHATO4B5MBwgcFMC4xLjTIBxCACAA&sclient=gws-wiz-img"
DOWNLOAD_DIR = "pictures/pomeranian"
MAX_DOWNLOADS = 100  # Maximum number of images to download (set to None for no limit)
########################################################

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def setup_download_directory():
    """Create download directory if it doesn't exist"""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        print(f"Created directory: {DOWNLOAD_DIR}")
    return DOWNLOAD_DIR

def download_image(img_url, save_path):
    """Download an image from URL and save it"""
    try:
        response = requests.get(img_url, headers=HEADERS, timeout=10, stream=True)
        response.raise_for_status()
        
        # Check if it's actually an image
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            return False
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")
        return False

def get_filename_from_url(url, index=None):
    """Extract filename from URL"""
    parsed = urlparse(url)
    path = parsed.path
    
    # Try to get filename from path
    filename = os.path.basename(path)
    
    # If no extension or invalid, create one
    if not filename or '.' not in filename:
        # Try to get from query parameters or create default
        filename = f"{prefix}_{index}.jpg" if index else "image.jpg"
    
    # Clean filename
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    return filename

def extract_images_from_selenium(driver):
    """Extract image URLs using Selenium"""
    image_urls = set()
    
    try:
        # Wait for page to load
        print("Waiting for page to load...")
        time.sleep(5)
        
        # Scroll to load more content (lazy loading)
        print("Scrolling to load more images...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_scrolls = 15  # Increased for more content
        
        while scroll_attempts < max_scrolls:
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Check if new content loaded
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            scroll_attempts += 1
        
        # Find all img tags
        img_elements = driver.find_elements(By.TAG_NAME, "img")
        print(f"Found {len(img_elements)} img elements")
        
        for img in img_elements:
            # Try different attributes for image URLs
            src = img.get_attribute("src")
            data_src = img.get_attribute("data-src")
            data_lazy_src = img.get_attribute("data-lazy-src")
            srcset = img.get_attribute("srcset")
            
            for url in [src, data_src, data_lazy_src]:
                if url and url.startswith(('http://', 'https://')):
                    # Filter out small icons, logos, etc.
                    if any(skip in url.lower() for skip in ['icon', 'logo', 'avatar', 'button', 'spinner', 'placeholder']):
                        continue
                    # Google Arts & Culture image URLs often contain 'lh3.googleusercontent.com'
                    if 'googleusercontent.com' in url or 'gstatic.com' in url:
                        # Try to get higher resolution
                        if '=w' in url:
                            # Replace with larger width
                            url = re.sub(r'=w\d+', '=w2048', url)
                        elif '=s' in url:
                            # Replace with larger size
                            url = re.sub(r'=s\d+', '=s2048', url)
                    image_urls.add(url)
            
            # Parse srcset if available
            if srcset:
                srcset_urls = re.findall(r'(https?://[^\s,]+)', srcset)
                for url in srcset_urls:
                    if any(skip in url.lower() for skip in ['icon', 'logo', 'avatar']):
                        continue
                    if 'googleusercontent.com' in url or 'gstatic.com' in url:
                        if '=w' in url:
                            url = re.sub(r'=w\d+', '=w2048', url)
                        elif '=s' in url:
                            url = re.sub(r'=s\d+', '=s2048', url)
                    image_urls.add(url)
        
        # Also look for background images in style attributes
        elements_with_bg = driver.find_elements(By.XPATH, "//*[@style]")
        for elem in elements_with_bg:
            style = elem.get_attribute("style")
            if style and "background-image" in style:
                # Extract URL from background-image: url(...)
                urls = re.findall(r'url\(["\']?([^"\']+)["\']?\)', style)
                for url in urls:
                    if url.startswith(('http://', 'https://')):
                        if 'googleusercontent.com' in url or 'gstatic.com' in url:
                            if '=w' in url:
                                url = re.sub(r'=w\d+', '=w2048', url)
                            elif '=s' in url:
                                url = re.sub(r'=s\d+', '=s2048', url)
                        image_urls.add(url)
        
        # Try to find images in script tags (JSON data) - Google Arts & Culture uses JSON
        script_tags = driver.find_elements(By.TAG_NAME, "script")
        for script in script_tags:
            script_content = script.get_attribute("innerHTML")
            if script_content:
                # Look for Google Arts & Culture image URLs
                # Pattern: googleusercontent.com URLs
                urls = re.findall(r'https?://[^\s"\'<>]+googleusercontent\.com[^\s"\'<>]+', script_content)
                for url in urls:
                    if any(skip in url.lower() for skip in ['icon', 'logo', 'avatar']):
                        continue
                    # Try to get higher resolution
                    if '=w' in url:
                        url = re.sub(r'=w\d+', '=w2048', url)
                    elif '=s' in url:
                        url = re.sub(r'=s\d+', '=s2048', url)
                    image_urls.add(url)
                
                # Also look for standard image URLs
                standard_urls = re.findall(r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|gif|webp)', script_content, re.IGNORECASE)
                for url in standard_urls:
                    if any(skip in url.lower() for skip in ['icon', 'logo', 'avatar']):
                        continue
                    image_urls.add(url)
        
    except Exception as e:
        print(f"Error extracting images with Selenium: {e}")
    
    return image_urls

def extract_images_from_api():
    """Try to extract images from API calls"""
    image_urls = set()
    
    # Google Arts & Culture might use API endpoints
    # Try to find the collection API endpoint
    collection_id = "the-international-museum-of-children"
    
    # Common API patterns for Google Arts & Culture
    api_urls = [
        f"https://artsandculture.google.com/api/collection/{collection_id}",
        f"https://www.google.com/culturalinstitute/api/collection/{collection_id}",
    ]
    
    for api_url in api_urls:
        try:
            response = requests.get(api_url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Recursively search for image URLs in JSON
                def find_urls(obj):
                    if isinstance(obj, dict):
                        for value in obj.values():
                            find_urls(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            find_urls(item)
                    elif isinstance(obj, str) and obj.startswith(('http://', 'https://')):
                        if any(ext in obj.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                            image_urls.add(obj)
                
                find_urls(data)
        except Exception as e:
            continue
    
    return image_urls

def main():
    """Main function to download images"""
    print(f"Starting image download from: {URL}")
    
    # Setup directory
    download_dir = setup_download_directory()
    
    all_image_urls = set()
    
    # Method 1: Try with Selenium (for dynamic content)
    print("\n=== Method 1: Using Selenium ===")
    if SELENIUM_AVAILABLE:
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'user-agent={HEADERS["User-Agent"]}')
            chrome_options.add_argument('--window-size=1920,1080')
            
            # Try to use webdriver-manager if available
            if WEBDRIVER_MANAGER_AVAILABLE:
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                driver = webdriver.Chrome(options=chrome_options)
            
            driver.get(URL)
            
            selenium_urls = extract_images_from_selenium(driver)
            all_image_urls.update(selenium_urls)
            print(f"Found {len(selenium_urls)} images with Selenium")
            
            driver.quit()
        except Exception as e:
            print(f"Selenium method failed: {e}")
            print("Note: Selenium requires ChromeDriver. You can:")
            print("  1. Install selenium: pip install selenium")
            print("  2. Download ChromeDriver from: https://chromedriver.chromium.org/")
            print("  3. Or use webdriver-manager: pip install webdriver-manager")
    else:
        print("Selenium not available. Install with: pip install selenium")
        print("For automatic ChromeDriver management: pip install webdriver-manager")
    
    # Method 2: Try with requests + BeautifulSoup
    print("\n=== Method 2: Using requests + BeautifulSoup ===")
    try:
        response = requests.get(URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all img tags
        img_tags = soup.find_all('img')
        for img in img_tags:
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            if src:
                # Convert relative URLs to absolute
                full_url = urljoin(URL, src)
                if full_url.startswith(('http://', 'https://')):
                    all_image_urls.add(full_url)
        
        # Find background images
        elements_with_bg = soup.find_all(style=re.compile(r'background-image'))
        for elem in elements_with_bg:
            style = elem.get('style', '')
            urls = re.findall(r'url\(["\']?([^"\']+)["\']?\)', style)
            for url in urls:
                full_url = urljoin(URL, url)
                if full_url.startswith(('http://', 'https://')):
                    all_image_urls.add(full_url)
        
        print(f"Found {len(img_tags)} img tags with BeautifulSoup")
    except Exception as e:
        print(f"BeautifulSoup method failed: {e}")
    
    # Method 3: Try API extraction
    print("\n=== Method 3: Trying API endpoints ===")
    api_urls = extract_images_from_api()
    all_image_urls.update(api_urls)
    print(f"Found {len(api_urls)} images from API")
    
    # Filter and clean URLs
    print(f"\nTotal unique image URLs found: {len(all_image_urls)}")
    
    # Filter out non-image URLs and small images
    filtered_urls = []
    for url in all_image_urls:
        # Skip icons, logos, etc.
        if any(skip in url.lower() for skip in ['icon', 'logo', 'avatar', 'button', 'spinner', 'placeholder']):
            continue
        # Prefer high-resolution images
        if '=w' in url or '=s' in url:
            # Try to get higher resolution
            url = url.split('=')[0] + '=w2048'  # Request larger size
        filtered_urls.append(url)
    
    print(f"Filtered to {len(filtered_urls)} relevant images")
    
    # Apply download limit if specified
    if MAX_DOWNLOADS is not None and MAX_DOWNLOADS > 0:
        original_count = len(filtered_urls)
        filtered_urls = filtered_urls[:MAX_DOWNLOADS]
        print(f"Limited to {MAX_DOWNLOADS} images (from {original_count} found)")
    
    # Download images
    print(f"\n=== Downloading images to '{download_dir}' ===")
    downloaded_count = 0
    failed_count = 0
    total_to_download = len(filtered_urls)
    
    for idx, img_url in enumerate(filtered_urls, 1):
        filename = get_filename_from_url(img_url, idx)
        save_path = os.path.join(download_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(save_path):
            print(f"[{idx}/{total_to_download}] Skipping (exists): {filename}")
            downloaded_count += 1
            continue
        
        print(f"[{idx}/{total_to_download}] Downloading: {filename}")
        if download_image(img_url, save_path):
            downloaded_count += 1
            print(f"  ✓ Saved: {save_path}")
        else:
            failed_count += 1
            print(f"  ✗ Failed: {img_url}")
        
        # Be polite - small delay between requests
        time.sleep(0.5)
    
    print(f"\n=== Download Complete ===")
    print(f"Successfully downloaded: {downloaded_count}")
    print(f"Failed: {failed_count}")
    if MAX_DOWNLOADS is not None:
        print(f"Download limit: {MAX_DOWNLOADS}")
    print(f"Total processed: {len(filtered_urls)}")

if __name__ == "__main__":
    main()
