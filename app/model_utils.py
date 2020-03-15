from app import app
import pandas as pd
import requests
import json
import random
import argparse
import torch
from torch_geometric.datasets import MovieLens

from .pgat_recsys import PGATRecSys
from .utils import get_folder_path
from .apikey import apikey

default_poster_src = 'https://www.nehemiahmfg.com/wp-content/themes/dante/images/default-thumb.png'

########################## Define arguments ##########################
parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument("--dataset", type=str, default='movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
parser.add_argument("--directed", type=str, default=False, help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--step_length", type=int, default=2, help="")
parser.add_argument("--train_ratio", type=float, default=None, help="")
parser.add_argument("--debug", default=0.01, help="")

# Model params
parser.add_argument("--heads", type=int, default=4, help="")
parser.add_argument("--dropout", type=float, default=0.6, help="")
parser.add_argument("--emb_dim", type=int, default=16, help="")
parser.add_argument("--repr_dim", type=int, default=16, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")

# Device params
parser.add_argument("--device", type=str, default='cpu', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")

args = parser.parse_args()

data_folder, weights_folder, logger_folder = get_folder_path(args.dataset + args.dataset_name)

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


########################## Setup Device ##########################
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)


########################## Define parameters ##########################
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'directed': args.directed, 'num_core': args.num_core, 'step_length': args.step_length, 'train_ratio': args.train_ratio,
    'debug': args.debug
}
model_args = {
    'heads': args.heads, 'emb_dim': args.emb_dim,
    'repr_dim': args.repr_dim, 'dropout': args.dropout,
    'hidden_size': args.hidden_size, 'model_path': weights_folder
}
device_args = {'debug': args.debug, 'device': device, 'gpu_idx': args.gpu_idx}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('device_args params: {}'.format(device_args))

recsys = PGATRecSys(num_recs=10, dataset_args=dataset_args, model_args=model_args, device_args=device_args)

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
