import pandas as pd
import requests
import json

default_poster_src = 'https://www.nehemiahmfg.com/wp-content/themes/dante/images/default-thumb.png'
APIKEY = 'ca2a706a'


def get_movie_name_for_id(i):
    i = int(i)
    movies = pd.read_csv('app/ml-1m/movies.dat', sep='::', engine='python')
    movie_name = movies['MovieName'][i]
    return movie_name


def get_movie_poster(movie_df):
    movie_title = movie_df['title']
    movie_year = movie_df['year']

    try:
        movie_url = 'http://www.omdbapi.com/?' + 't=' + movie_title + '&apikey=' + APIKEY
        r = requests.get(movie_url)
        movie_info_dict = json.loads(r.text)

    except:
        try:
            movie_url = 'http://www.omdbapi.com/?' + 't=' + movie_title + '&y=' + str(movie_year) + '&apikey=' + APIKEY
            r = requests.get(movie_url)
            movie_info_dict = json.loads(r.text)
        except:
            try:
                movie_url = 'http://www.omdbapi.com/?' + 't=' + movie_title + '&y=' + str(
                    movie_year - 1) + '&apikey=' + APIKEY
                r = requests.get(movie_url)
                movie_info_dict = json.loads(r.text)
            except:
                try:
                    movie_url = 'http://www.omdbapi.com/?' + 't=' + movie_title + '&y=' + str(
                        movie_year + 1) + '&apikey=' + APIKEY
                    r = requests.get(movie_url)
                    movie_info_dict = json.loads(r.text)
                except:
                    movie_info_dict = dict()

    poster = movie_info_dict.get('Poster', default_poster_src)
    return poster
