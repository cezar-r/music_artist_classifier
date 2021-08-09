import requests

scraper_url = 'http://api.scraperapi.com?api_key=<API KEY HERE>&url='
base_url = 'https://www.listchallenges.com/500-random-musical-artists'

artists = []

def scrape_artists():
	"""
	Function that scrapes listchallenges.com and collects various artist names
	This file should not be run on its own
	"""
	for i in range(1, 14):
		if i > 1:
			response = requests.get(base_url + f'/list/{i}')
		else:
			response = requests.get(base_url)
		html = response.text
		html = html.split('class="item-name">\r\n\t\t\t\t\t\t\t\t\t\t')
		for div in html[1:]:
			corr_div = div.split('\r\n\t')
			name = corr_div[0]
			if name.lower() not in artists:
				artists.append(name.lower())


scrape_artists()
