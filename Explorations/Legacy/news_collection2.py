import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import os

# List of ticker symbols to analyze
TICKER_SYMBOLS = [
    'AAPL',  # Apple
    'GOOGL', # Google
    'MSFT',  # Microsoft
    'AMZN',  # Amazon
    # Add more tickers here as needed
]

def get_stock_news(symbol, num_articles=10):
    """
    Fetch recent news articles for a given stock symbol.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
        num_articles (int): Number of articles to retrieve (default: 10)
    
    Returns:
        list: List of dictionaries containing article information
    """
    # Convert symbol to uppercase
    symbol = symbol.upper()
    
    # Yahoo Finance news URL
    url = f"https://finance.yahoo.com/quote/{symbol}/news"
    
    # Headers to mimic browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Make the request
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find news articles
        articles = []
        news_items = soup.find_all('div', {'class': 'Py(14px)'})
        
        for item in news_items[:num_articles]:
            try:
                # Find the article link
                link_element = item.find('a')
                if not link_element:
                    continue
                    
                title = link_element.text.strip()
                link = 'https://finance.yahoo.com' + link_element['href'] if link_element['href'].startswith('/') else link_element['href']
                
                # Get article source and time
                source_time = item.find('div', {'class': 'C(#959595)'})
                if source_time:
                    source_time_text = source_time.text.split('·')
                    source = source_time_text[0].strip()
                    pub_time = source_time_text[1].strip() if len(source_time_text) > 1 else 'N/A'
                else:
                    source = 'Unknown'
                    pub_time = 'N/A'
                
                # Get article summary if available
                summary_element = item.find('p')
                summary = summary_element.text.strip() if summary_element else 'No summary available'
                
                articles.append({
                    'title': title,
                    'source': source,
                    'published': pub_time,
                    'summary': summary,
                    'link': link
                })
                
            except Exception as e:
                print(f"Error processing article: {str(e)}")
                continue
                
        return articles
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {symbol}: {str(e)}")
        return []

def save_to_csv(articles, symbol):
    """
    Save articles to a CSV file.
    
    Args:
        articles (list): List of article dictionaries
        symbol (str): Stock symbol for filename
    """
    if not articles:
        print(f"No articles to save for {symbol}")
        return
        
    # Create 'news_data' directory if it doesn't exist
    os.makedirs('news_data', exist_ok=True)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(articles)
    filename = f"news_data/{symbol}_news_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Saved news data for {symbol} to {filename}")

def main():
    print(f"Starting news collection for {len(TICKER_SYMBOLS)} symbols...")
    
    for symbol in TICKER_SYMBOLS:
        print(f"\nProcessing {symbol}...")
        
        # Get the news articles
        articles = get_stock_news(symbol)
        
        # Save to CSV
        save_to_csv(articles, symbol)
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(2)
    
    print("\nNews collection completed!")

if __name__ == "__main__":
    main()