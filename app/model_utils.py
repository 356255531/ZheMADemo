from app import app
import pandas as pd
import requests
import json

# from .mpagat import MPAGATRecsys
from .emf import EMFRecsys
from .apikey import apikey

default_poster_src = 'https://www.nehemiahmfg.com/wp-content/themes/dante/images/default-thumb.png'

# save id selected by users
current_user_id = ''
iid_list = []
iid_list2 = []
iid_list3 = []
demographic_info = ()
rs_proportion = {'IUI':3,
                 'UIU':3,
                 'IUDD':2,
                 'UICC':2,
                 'SUM':10}

recsys = EMFRecsys()

refresh_value = 0


# Model Util Functions

def get_top_movie_ids(n):
    movie_df = recsys.get_top_n_popular_items(n)
    iids = [iid for iid in movie_df.iid.values]
    return iids


def get_movie_name_for_id(i):
    i = int(i)
    movies = pd.read_csv('app/ml-1m/movies.dat', sep='::', engine='python')
    movie_name = movies['MovieName'][i]
    return movie_name


def get_movie_poster(movie_df):
    movie_title = movie_df['title']
    movie_year = movie_df['year']

    movie_url = "http://www.omdbapi.com/?t=%s&y=%i&apikey=%s" % (movie_title, movie_year, apikey)
    movie_url_no_year = "http://www.omdbapi.com/?t=%s&apikey=%s" % (movie_title, apikey)

    r = requests.get(movie_url)
    response_text = json.loads(r.text)
    try:
        movie_info_dic = response_text
        poster = movie_info_dic['Poster']
        if poster == 'N/A':
            return default_poster_src
        else:
            return poster

    except:
        response_value = response_text['Response']
        if 'False' == response_value:
            r2 = requests.get(movie_url_no_year)
            movie_info_dic2 = json.loads(r2.text)
            try:
                poster2 = movie_info_dic2['Poster']
                if poster2 == 'N/A':
                    return default_poster_src
                else:
                    return poster2
            except:
                return default_poster_src


def run_adaptation_model(user_id,proportion,round_number):
    # get type and score data from sqlite3 database
    explanation_type_and_score_list = select_explanation_type_and_score(user_id,round_number)

    # calculate average score
    average_score_of_sum = sum([int(type_score[1]) for type_score in explanation_type_and_score_list]) / len(explanation_type_and_score_list)

    iui_score_list = [int(type_score[1]) for type_score in explanation_type_and_score_list if type_score[0] == 'IUI']
    if len(iui_score_list) != 0:
        average_score_of_iui = sum(iui_score_list)/len(iui_score_list)
    else:
        # if this exp type did not appear in previous round,
        # add one in next round to test whether user may like it
        average_score_of_iui = average_score_of_sum + 1

    uiu_score_list = [int(type_score[1]) for type_score in explanation_type_and_score_list if type_score[0] == 'UIU']
    if len(uiu_score_list) != 0:
        average_score_of_uiu = sum(uiu_score_list) / len(uiu_score_list)
    else:
        average_score_of_uiu = average_score_of_sum + 1

    iudd_score_list = [int(type_score[1]) for type_score in explanation_type_and_score_list if type_score[0] == 'IUDD']
    if len(iudd_score_list) != 0:
        average_score_of_iudd = sum(iudd_score_list) / len(iudd_score_list)
    else:
        average_score_of_iudd = average_score_of_sum + 1

    uicc_score_list = [int(type_score[1]) for type_score in explanation_type_and_score_list if type_score[0] == 'UICC']
    if len(uicc_score_list) != 0:
        average_score_of_uicc = sum(uicc_score_list) / len(uicc_score_list)
    else:
        average_score_of_uicc = average_score_of_sum + 1

    # Adaptation Model
    # create new proportion
    new_rs_proportion = {'IUI': proportion['IUI'] + round(average_score_of_iui - average_score_of_sum),
                         'UIU': proportion['UIU'] + round(average_score_of_uiu - average_score_of_sum),
                         'IUDD': proportion['IUDD'] + round(average_score_of_iudd - average_score_of_sum),
                         'UICC': proportion['UICC'] + round(average_score_of_uicc - average_score_of_sum), 'SUM': 10}
    if new_rs_proportion['IUI'] < 0:
        new_rs_proportion['IUI'] = 0
    if new_rs_proportion['IUI'] > 10:
        new_rs_proportion['IUI'] = 10

    if new_rs_proportion['UIU'] < 0:
        new_rs_proportion['UIU'] = 0
    if new_rs_proportion['UIU'] > 10:
        new_rs_proportion['UIU'] = 10

    if new_rs_proportion['IUDD'] < 0:
        new_rs_proportion['IUDD'] = 0
    if new_rs_proportion['IUDD'] > 10:
        new_rs_proportion['IUDD'] = 10

    if new_rs_proportion['UICC'] < 0:
        new_rs_proportion['UICC'] = 0
    if new_rs_proportion['UICC'] > 10:
        new_rs_proportion['UICC'] = 10

    while new_rs_proportion['IUI'] + new_rs_proportion['UIU'] + new_rs_proportion['IUDD'] + new_rs_proportion['UICC'] > new_rs_proportion['SUM']:
        new_rs_proportion['IUI'] -= 1

    while new_rs_proportion['IUI'] + new_rs_proportion['UIU'] + new_rs_proportion['IUDD'] + new_rs_proportion['UICC'] < new_rs_proportion['SUM']:
        new_rs_proportion['IUI'] += 1

    return new_rs_proportion
