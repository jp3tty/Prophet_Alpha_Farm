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
    Fetch recent news articles for a given stock symbol using Finviz instead of Yahoo Finance.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
        num_articles (int): Number of articles to retrieve (default: 10)
    
    Returns:
        list: List of dictionaries containing article information
    """
    symbol = symbol.upper()
    url = f"https://finviz.com/quote.ashx?t={symbol}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    try:
        print(f"Fetching news for {symbol}...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the news table
        news_table = soup.find('table', {'class': 'news-table'})
        if not news_table:
            print(f"No news table found for {symbol}")
            return []
        
        articles = []
        rows = news_table.findAll('tr')
        
        for row in rows[:num_articles]:
            try:
                # Get date
                date_cell = row.find('td')
                if not date_cell:
                    continue
                    
                # Get article link and title
                link_element = row.find('a')
                if not link_element:
                    continue
                
                title = link_element.text.strip()
                link = link_element['href']
                date_text = date_cell.text.strip()
                
                articles.append({
                    'date': date_text,
                    'title': title,
                    'link': link,
                    'symbol': symbol
                })
                
            except Exception as e:
                print(f"Error processing article row: {str(e)}")
                continue
        
        print(f"Found {len(articles)} articles for {symbol}")
        return articles
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {symbol}: {str(e)}")
        return []
    except Exception as e:
        print(f"Unexpected error for {symbol}: {str(e)}")
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
        return False
        
    try:
        # Create 'news_data' directory if it doesn't exist
        os.makedirs('news_data', exist_ok=True)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(articles)
        filename = f"news_data/{symbol}_news_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        
        # Verify file was created
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            print(f"Successfully saved {len(articles)} articles for {symbol} to {filename}")
            return True
        else:
            print(f"Failed to save articles for {symbol} - file is empty or not created")
            return False
            
    except Exception as e:
        print(f"Error saving articles for {symbol}: {str(e)}")
        return False

def main():
    print(f"Starting news collection for {len(TICKER_SYMBOLS)} symbols...")
    successful_saves = 0
    
    for symbol in TICKER_SYMBOLS:
        print(f"\nProcessing {symbol}...")
        
        # Get the news articles
        articles = get_stock_news(symbol)
        
        # Save to CSV
        if save_to_csv(articles, symbol):
            successful_saves += 1
        
        # Add a delay between requests
        time.sleep(2)
    
    print(f"\nNews collection completed!")
    print(f"Successfully saved news for {successful_saves} out of {len(TICKER_SYMBOLS)} symbols")
    
    # List all saved files
    if successful_saves > 0:
        print("\nSaved files:")
        for filename in os.listdir('news_data'):
            filepath = os.path.join('news_data', filename)
            size = os.path.getsize(filepath)
            print(f"- {filename} ({size} bytes)")

if __name__ == "__main__":
    main()