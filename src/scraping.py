import requests
import json
import re 
import urllib.request
from bs4 import BeautifulSoup
from cmd_colors import Colors as c
from artists import artists 
import time

last_fm_api_key = 'e34c6de6772e623c5f8fac80e4752db4'
last_fm_url = 'http://ws.audioscrobbler.com'
scraper_url = 'http://api.scraperapi.com?api_key=800a43e7386a5a4f5801c8762c3a4aab&url='
bad_chars = ['?', '!', '&', ':', ';', '</i>', '<i>', '[', ']', ',', '(', ')', 'INTRO:', 'Chorus:', 'CHORUS:', 'Verse:', 'Bridge:', 'Hook:', 'HOOK:', '—', '–', '�', 'á', 'à', 'â', 'ä', 'ç', 'é', 'è', 'ë', 'í', 'ì', 'ï', 'ì', 'Ò', 'ó', 'ò', 'ö', 'ú', 'ù', 'û', 'ü', 'ß', 'ñ', '…', '’', '‘', '“', '”', '¿', f'\n', f'\n\n', "'", '"', '...', '\r', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
stop_words = 'a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,guy,had,has,have,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,should,since,so,some,something,than,that,the,their,them,then,there,theres,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,wont,yet,you,your'.split(',')


class LyricScraper:


    def __init__(self, artists, verbose = True):

        self.artists = artists
        self.verbose = verbose
    

    def _get_songs(self, artist, retries = 0):
        if self.verbose:
            print(f"Getting songs for {artist}")
        songs = []
        response = requests.get(last_fm_url + f'/2.0/?method=artist.gettoptracks&artist={artist}&api_key={last_fm_api_key}&format=json')
        try:
            json_text = json.loads(response.text)
        except Exception as e:
            print(str(e) + '\nRetrying in 10 seconds')
            time.sleep(10)
            retries += 1
            if retries < 4:
                self._get_genres(artist, song, retries)
            else:
                return None
        try:
            tracks = json_text['toptracks']
        except Exception as e:
            print(str(e) + '\n Couldnt find songs for: ' + artist)
            return None
        track = tracks['track']
        for json_obj in track:
            song_name = json_obj['name']
            if self.verbose:
                pass
                # print(song_name)
            songs.append(song_name)
        return songs
        # for song-genre for artist
    

    def _get_genres(self, artist, song, retries = 0):
        response = requests.get(last_fm_url + f'/2.0/?method=track.getInfo&api_key={last_fm_api_key}&artist={artist}&track={song}&format=json')
        try:
            json_text = json.loads(response.text)
        except Exception as e:
            print(str(e) + '\nRetrying in 10 seconds Retry =', retries)
            time.sleep(10)
            retries += 1
            if retries < 4:
                genre = self._get_genres(artist, song, retries)
                return genre
            else:
                print('returning none')
                return None
        # print(json.dumps(json_text, indent=4, sort_keys=True))
        if 'error' == list(json_text.keys())[0]:
            print(f"Couldn't find song")
            print(json_text)
            return None

        track = json_text['track']
        toptags = track['toptags']
        if toptags == '':
            if self.verbose:
                print(f"Empty toptags, no genre")
            return None

        tags = toptags['tag']
        for tag in tags:
            if type(tag) == str:
                if self.verbose:
                    print(f"tag was a string, skipping")
                return None 

            genre = tag['name']
            if genre != artist:
                print(f"{song.title()}: {genre}")
                return genre


    def _clean_song_name(self, name):
        name = name.split(' ')
        correct_name = ''
        for word in name:
            if len(word) < 1:
                continue
            if word[0] != '(':
                correct_name += word + ' '
            else:
                break
        return correct_name[:-1]


    def _get_lyrics(self, artist, song):
        # scrape genius instead
        song_title = self._clean_song_name(song)
        artist = artist.lower() 
        song_title = song_title.lower() 
        # remove all except alphanumeric characters from artist and song_title 
        artist = re.sub('[^A-Za-z0-9]+', "", artist) 
        song_title = re.sub('[^A-Za-z0-9]+', "", song_title) 
        if artist.startswith("the"):    # remove starting 'the' from artist e.g. the who -> who 
            artist = artist[3:] 
        if artist == 'playboicarti':
            artist = 'playboi-carti'
        url = "http://azlyrics.com/lyrics/"+artist+"/"+song_title+".html"
        if self.verbose:
            print(f"{url}", end="")
        new_url = scraper_url + url 
        try: 
            content = urllib.request.urlopen(new_url).read() 
            soup = BeautifulSoup(content, 'html.parser') 
            lyrics = str(soup) 
            # lyrics lies between up_partition and down_partition 
            up_partition = '<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->' 
            down_partition = '<!-- MxM banner -->' 
            lyrics = lyrics.split(up_partition)[1] 
            lyrics = lyrics.split(down_partition)[0] 
            lyrics = lyrics.replace('<br/>','').replace('</br>','').replace('</div>','').strip().replace('\n', ' ')
            return lyrics, song
        except Exception as e: 
            return "\nException occurred \n" +str(e)  
        # get lyrics for song
    

    def _clean_lyrics(self, lyric_song_tup):
        lyric = lyric_song_tup[0]
        song = lyric_song_tup[1]
        for char in bad_chars:
            lyric = lyric.replace(char, "")
            lyric = ' '.join([i.lower()  for i in lyric.split(' ') if len(i) > 1 and i not in stop_words])
        return lyric
        # clean lyric that is passed in


    def _file_has(self, label, song = None, file_name = 'test'):
        file = open(f"../data/{file_name}.txt", "r", encoding='utf-8').readlines()
        if song is None:
            count = 0
            for line in file:
                if count == 40:
                    return True
                if line.split('|')[0] == label:
                    count += 1
            return False
        else:
            for line in file:
                if line.split('|')[1] == song and line.split('|')[0] == label:
                    return True
            return False


    def write(self, outfile="test"):
        for artist in self.artists:
            if self._file_has(artist):
                if self.verbose:
                    print(f"Duplicate: {artist}, skipping")
                continue
            songs = self._get_songs(artist)
            if songs is None:
                continue
            for song in songs:
                if self._file_has(artist, song):
                    if self.verbose:
                        print(f"Skipping {song}")
                    continue
                lyric = self._clean_lyrics(self._get_lyrics(artist, song))
                if self.verbose:
                    print(f"{songs.index(song) + 1}/{len(songs)}")
                if len(lyric.split(' ')) < 30:
                    print("Couldn't find song or song lyrics, skipping", lyric)
                    continue    
                genre = self._get_genres(artist, song)
                try:
                    with open(f"../data/{outfile}.txt", "a", encoding='utf-8') as f:
                        f.write(f"{artist}|{song}|{lyric}|{genre}\n")
                        if self.verbose:
                            print(f'"{song.title()}" by {artist.title()} has been written to file')
                except Exception as e:
                    print("\nException occurred \n" + str(e))
                    if self.verbose:
                        print(lyric)
                print(' ')
            if self.verbose:
                print('\n\n')


    def add_artist(self, artist, outfile = 'test'):
        if not self._has_file(artist, col = 'artist'):
            dup = self.artists.copy()
            self.artists = [artist]
            self.write(outfile)
            self.artists = dup


    def add_song(self, artist, song, outfile = 'test'):
        if self._file_has(artist, song):
            return 'Song already in list'
        lyric = self._clean_lyrics(self._get_lyrics(artist, song))
        if len(lyric.split(' ')) < 30:
            print("Couldn't find song, skipping", lyric)
            return    
        genre = self._get_genres(artist, song)
        try:
            with open(f"../data/{outfile}.txt", "a", encoding='utf-8') as f:
                f.write(f"{artist}|{song}|{lyric}|{genre}\n")
                if self.verbose:
                    print(f'"{song.title()}" by {artist.title()} has been written to file')
        except Exception as e:
            print("\nException occurred \n" + str(e))
            if self.verbose:
                print(lyric)



if __name__ == '__main__':
    s = LyricScraper(artists)
    s.write()


'''
TODO
'''