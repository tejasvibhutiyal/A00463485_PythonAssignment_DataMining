# crypto_utils.py

import requests
import time
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv('COINGECKO_API_KEY')  # Assuming your API key is stored in an environment variable
HEADERS = {'Authorization': f'Bearer {API_KEY}'} if API_KEY else {}

def make_api_request(url, params=None, max_retries=3, backoff_factor=2):
    """Make an API request with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            # Check if the error is due to rate limiting or other retry-able errors
            if response.status_code in {429, 503}:
                # Exponential backoff
                time.sleep((backoff_factor ** attempt) * 0.1)
            else:
                raise http_err
        except requests.exceptions.RequestException as err:
            if attempt == max_retries - 1:
                raise err
            time.sleep((backoff_factor ** attempt) * 0.1)

from functools import lru_cache

@lru_cache(maxsize=1)
def fetch_all_coins_list():
    """Fetch the list of all cryptocurrencies. Cached to avoid re-fetching."""
    url = 'https://api.coingecko.com/api/v3/coins/list'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

def fetch_historical_data(coin_id, days=365):
    """Fetch historical price data for a cryptocurrency."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    params = {
        'vs_currency': 'usd',
        'from': int(start_date.timestamp()),
        'to': int(end_date.timestamp())
    }

    response = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range", params=params)
    if response.status_code == 200 and 'prices' in response.json():
        return response.json()['prices']
    else:
        # Return None or handle error accordingly
        return None