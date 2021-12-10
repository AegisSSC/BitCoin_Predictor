# https://tommycc.medium.com/web-scraping-crypto-prices-with-python-41072ea5b5bf
from bs4 import BeautifulSoup
import requests
import pandas as pd 
import json
import time

cmc = requests.get('https://coinmarketcap.com')
soup = BeautifulSoup(cmc.content, 'html.parser')

print("The Title of the HTML is: " )
print(soup.title)
print("The raw data formatted is: " )
print(soup.prettify())

#Get all of the json files associated with the ids
data = soup.find('script', id = '__NEXT_DATA__', type = "application/json")

coins = {}

#remove the script tags using data.contents[0]
coin_data = json.loads(data.contents[0])
listings = coin_data['props']['initialState']['cryptocurrency']['listingLatest']['data']

#for every id in the listing, create an id-slug pairing to use to grab the crypto's information
for i in listings:
    coins[str(i['id'])] = i['slug']


for i in coins:
    page = requests.get(f'https://coinmarketcap.com/currencies/{coins[i]}/historical-data/?start=20200101&end=20200630')
    soup = BeautifulSoup(page.content, 'html.parser')
    data = soup.find('script', id = "__NEXT_DATA__", type = "application/json")
    historical_data = json.loads(data.conents[0])
    quotes = historical_data['props']['initialState'] ['cryptocurrency']['ohlcvHistorical'][i]['quotes']

    market_cap = []
    volume = []
    timestamp = []
    name = []
    symbol = []
    slug = []

    for j in quotes:
        market_cap.append(j['quote']['USD']['market_cap'])
        volume.append(j['quote']['USD']['volume'])
        timestamp.append(j['quote']['USD']['timestamp'])
        name.append(info['name'])
        symbol.append(info['symbol'])
        slug.append(coins[i])

    datafile = pd.DataFrame(columns =['marketcap', 'volume', 'timestamp', 'name', 'symbol', 'slug'])
    datafile['marketcap'] = market_cap
    datafile['volume'] = volume
    datafile['timetamp'] = timestamp
    datafile['name'] = name
    datafile['symbol'] = symbol
    datafile['slug'] = slug
datafile.to_csv('cryptoes.csv', index = False)
