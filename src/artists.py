import requests

scraper_url = 'http://api.scraperapi.com?api_key=800a43e7386a5a4f5801c8762c3a4aab&url='
base_url = 'https://www.listchallenges.com/500-random-musical-artists'

artists = []

def scrape_artists():
	for i in range(8, 14):
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
