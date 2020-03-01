from flask import render_template, request
from app import app
import pandas as pd
import requests
# from django.http import HttpResponse
import json
import sqlite3
import os.path
import random
import argparse
import torch
from torch_geometric.datasets import MovieLens
from .pgat_recsys import PGATRecSys
from .utils import get_folder_path

parser = argparse.ArgumentParser()
default_poster_src = 'https://www.nehemiahmfg.com/wp-content/themes/dante/images/default-thumb.png'


########################## Define arguments ##########################
# Dataset params
parser.add_argument("--dataset", type=str, default='movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
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

########################## Define arguments ##########################
data_folder, weights_folder, logger_folder = get_folder_path(args.dataset + args.dataset_name)

########################## Setup Device ##########################
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

########################## Define parameters ##########################
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'num_core': args.num_core, 'step_length': args.step_length, 'train_ratio': args.train_ratio,
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

@app.template_global()
def generateIDs(n):
    movie_df = recsys.get_top_n_popular_items(n)
    iids = [iid for iid in movie_df.iid.values]
    return iids

@app.template_global()
def get_movie_name_withID(i):
    i = int(i)
    movies = pd.read_csv('app/ml-1m/movies.dat', sep='::', engine='python')
    movie_name = movies['MovieName'][i]
    return movie_name

@app.template_global()
def get_movie_poster_withID(i):

    # apikey = 'e760129c'
    # apikey = 'e44e5305'
    apikey = 'c32748ad'
    # apikey = '192c6b0e'
    movies = pd.read_csv('app/ml-1m/movies.dat', sep='::', engine='python')
    movie_name = get_movie_name_withID(i)
    movie_title = movie_name[0:-7]
    movie_title = movie_title.replace(' ','+')
    movie_year = movie_name[-5:-1]

    movie_url = "http://www.omdbapi.com/?" + "t=" + movie_title + "&y=" + movie_year + "&apikey=" + apikey
    movie_url_no_year = "http://www.omdbapi.com/?" + "t=" + movie_title + "&apikey=" + apikey
    # print('movie_url: ' + movie_url)

    r = requests.get(movie_url)
    # print(r.text)
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
        # print('response_value:'+response_value)
        # {"Response": "False", "Error": "Movie not found!"}
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


@app.template_global()
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


@app.template_global()
def save_explanation_score_tosqlite(user_id,movie_id,seen_status,explanation_type,explanation_score,user_study_round):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "MATDemo.db")
    connection = sqlite3.connect(db_path)

    cursor = connection.cursor()
    print("Opened database successfully")

    cursor.execute('create table if not exists EXP_SCORE (user_id,movie_id,seen_status,explanation_type,explanation_score,user_study_round)')
    params = (user_id,movie_id,seen_status,explanation_type,explanation_score,user_study_round)

    cursor.execute("INSERT INTO EXP_SCORE VALUES (?,?,?,?,?,?)",params)
    connection.commit()
    print("Records created successfully")

    connection.close()
    return 1

@app.template_global()
def save_question_result1_tosqlite(user_id,question_result1_list):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "MATDemo.db")
    connection = sqlite3.connect(db_path)

    cursor = connection.cursor()
    print("Opened database successfully")

    cursor.execute('create table if not exists QUESTION_RESULT1 (user_id,q1,q2,q3,q4,q5,q6,q7)')
    params = (user_id,
              question_result1_list[0],
              question_result1_list[1],
              question_result1_list[2],
              question_result1_list[3],
              question_result1_list[4],
              question_result1_list[5],
              question_result1_list[6])

    cursor.execute("INSERT INTO QUESTION_RESULT1 VALUES (?,?,?,?,?,?,?,?)", params)
    connection.commit()
    print("Records created successfully")

    connection.close()
    return 1

@app.template_global()
def save_question_result2_tosqlite(user_id,question_result2_list):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "MATDemo.db")
    connection = sqlite3.connect(db_path)

    cursor = connection.cursor()
    print("Opened database successfully")

    cursor.execute('create table if not exists QUESTION_RESULT2 (user_id,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13)')
    params = (user_id,
              question_result2_list[0],
              question_result2_list[1],
              question_result2_list[2],
              question_result2_list[3],
              question_result2_list[4],
              question_result2_list[5],
              question_result2_list[6],
              question_result2_list[7],
              question_result2_list[8],
              question_result2_list[9],
              question_result2_list[10],
              question_result2_list[11],
              question_result2_list[12])

    cursor.execute("INSERT INTO QUESTION_RESULT2 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", params)
    connection.commit()
    print("Records created successfully")

    connection.close()
    return 1

@app.template_global()
def select_explanation_type_and_score(user_id,round_number):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "MATDemo.db")
    connection = sqlite3.connect(db_path)

    cursor = connection.cursor()
    print("Opened database successfully")

    # import pdb
    # pdb.set_trace()

    params = (str(user_id),str(round_number))
    cursor.execute("SELECT explanation_type, explanation_score FROM EXP_SCORE WHERE user_id=? AND user_study_round=?",params)

    explanation_type_score_rows = cursor.fetchall()

    # for row in rows:
    #     print(row)

    connection.commit()
    print("Select data successfully")

    return explanation_type_score_rows;

@app.route('/')
@app.route('/index')
def index():
    user = {'username':'Zhe'}
    return render_template('explanation.html', title = 'Film Recommendation', user = user)

@app.route('/explanation')
def explanation():
    return render_template('explanation.html', title = 'Film Recommendation')

@app.route('/user_information')
def user_information():
    return render_template('user_information.html', title = 'Film Recommendation')

@app.route('/user_background')
def user_background():
    return render_template('user_background.html',title = 'Film Recommendation')

@app.route('/question_result_transfer',methods=['GET','POST'])
def question_result_transfer():
    if request.method == 'POST':
        user_id = request.values['user_id']
        global current_user_id
        current_user_id = user_id
        question_result_list = request.values['question_result_list']
        if question_result_list != '':
            save_question_result1_tosqlite(user_id,question_result_list)
            return 'success'

@app.route('/movie_preview')
def movie_preview():

    movie_ids = generateIDs(500)
    step = 6
    group_movieIDs = [movie_ids[i:i + step] for i in range(0, len(movie_ids), step)]
    click_count = refresh_value
    # import pdb
    # pdb.set_trace()

    return render_template('movie_preview.html',title = 'Film Recommendation', group_movieIDs = group_movieIDs, click_count = click_count)

@app.route('/refresh_count',methods=['GET','POST'])
def refresh_count():
    if request.method == 'POST':
        temp_refresh_value = request.values['refresh_value']
        global refresh_value
        if temp_refresh_value != '':
            refresh_value = int(temp_refresh_value)
            return 'success'

@app.route('/imgID_userinfo_transfer',methods=['GET','POST'])
def imgID_userinfo_transfer():
    global demographic_info
    # import pdb
    # pdb.set_trace()
    if request.method == 'POST':
        the_id = request.values['id']
        the_id = int(the_id)
        gender = request.values['gender']
        occupation = request.values['occupation']
        # user_age = request.values['user_age']

        demographic_info = (gender, occupation)
        iid_list.append(the_id)
        if len(iid_list) == 10:
            print('creating new user...')
        return 'success'
    else:
        return 'fail'


@app.route('/movie_degree')
def movie_degree():

    global iid_list
    global demographic_info
    global rs_proportion

    recsys.build_user(iid_list, demographic_info)
    print('new user created')

    df, exps = recsys.get_recommendations(rs_proportion)
    rec_movie_iids = df.iid.values

    return render_template('movie_degree.html',title = 'Film Recommendation',rec_movie_iids_and_explanations = zip(rec_movie_iids,exps))

@app.route('/movie_name_transfer',methods=['GET','POST'])
def movie_name_transfer():
    if request.method == 'POST':
        movie_id = request.values['movie_id']
        movie_name = get_movie_name_withID(movie_id)
        return movie_name

@app.route('/score_movie_transfer',methods=['GET','POST'])
def score_movie_transfer():
    global iid_list2
    if request.method == 'POST':
        user_id = request.values['user_id']
        movie_id = request.values['movie_id']
        seen_status = request.values['seen_status']
        explanation_type = request.values['explanation_type']
        score = request.values['score']
        user_study_round = "1"
        # save 10 {explanation_type:score}
        # run Adaptation Model to get new Explanation proportion
        # like {IUI:UIU:IUDD:UICC} = {1:2:3:4} sum=10
        print('get new data, user_id:{},movie_id:{},seen_status:{},explanation_type:{},score:{},user_study_round:{}'.format(user_id,movie_id,seen_status,explanation_type,score,user_study_round))


        # rs_proportion[explanation] += 1;

        save_explanation_score_tosqlite(user_id,movie_id,seen_status,explanation_type,score,user_study_round)

        the_id = int(movie_id)
        the_score = int(score)
        iid_list2.append(the_id)

        return 'success'
    else:
        return 'fail'

@app.route('/user_info_transfer',methods=['GET','POST'])
def user_info_transfer():
    if request.method == 'POST':

        user_id = int(request.values['user_id'])
        user_gender = recsys.data.users[0]['gender'][user_id]
        user_age = recsys.data.users[0]['age'][user_id]
        user_occ = recsys.data.users[0]['occupation'][user_id]

        ratings_iids = recsys.data.ratings[0]['iid']
        ratings_uids = recsys.data.ratings[0]['uid']
        # get iids of movie user has seen
        user_movie_iids = [iid for uid,iid in zip(ratings_uids,ratings_iids) if uid==user_id]
        user_movie_names = [get_movie_name_withID(iid) for iid in user_movie_iids]

        # import pdb
        # pdb.set_trace()

        user_info = {'user_id':str(user_id),'user_gender':user_gender,'user_age':user_age,'user_occ':user_occ,'user_movie_names':user_movie_names}
        return user_info

@app.route('/movie_degree2')
def movie_degree2():

    global iid_list2
    global demographic_info
    new_iids = recsys.base_iids + iid_list2

    # TODO:Send Adaptation Model parameter to build user in next round
    global current_user_id
    global rs_proportion

    new_rs_proportion = run_adaptation_model(current_user_id,rs_proportion,1)

    recsys.build_user(new_iids, demographic_info)
    df, exps = recsys.get_recommendations(new_rs_proportion)
    rec_movie_iids2 = df.iid.values


    # save new_rs_proportion to rs_proportion for next round
    rs_proportion = new_rs_proportion

    return render_template('movie_degree2.html',title = 'Film Recommendation',rec_movie_iids_and_explanations2 = zip(rec_movie_iids2,exps))

@app.route('/score_movie_transfer2',methods=['GET','POST'])
def score_movie_transfer2():
    global iid_list3
    if request.method == 'POST':
        user_id = request.values['user_id']
        movie_id = request.values['movie_id']
        seen_status = request.values['seen_status']
        explanation_type = request.values['explanation_type']
        score = request.values['score']
        user_study_round = "2"
        print('get new data, user_id:{},movie_id:{},seen_status:{},explanation_type:{},score:{},user_study_round:{}'.format(user_id,movie_id,seen_status,explanation_type,score,user_study_round))

        save_explanation_score_tosqlite(user_id,movie_id,seen_status,explanation_type,score,user_study_round)

        the_id = int(movie_id)
        the_score = int(score)
        iid_list3.append(the_id)

        return 'success'
    else:
        return 'fail'

@app.route('/movie_degree3')
def movie_degree3():

    global iid_list3
    global demographic_info
    new_iids = recsys.base_iids + iid_list3

    global current_user_id
    global rs_proportion

    new_rs_proportion = run_adaptation_model(current_user_id,rs_proportion,2)

    recsys.build_user(new_iids, demographic_info)
    df, exps = recsys.get_recommendations(new_rs_proportion)
    rec_movie_iids3 = df.iid.values

    # save new_rs_proportion to rs_proportion for next round

    rs_proportion = new_rs_proportion

    return render_template('movie_degree3.html',title = 'Film Recommendation',rec_movie_iids_and_explanations3 = zip(rec_movie_iids3,exps))

@app.route('/score_movie_transfer3',methods=['GET','POST'])
def score_movie_transfer3():
    if request.method == 'POST':
        user_id = request.values['user_id']
        movie_id = request.values['movie_id']
        seen_status = request.values['seen_status']
        explanation_type = request.values['explanation_type']
        score = request.values['score']
        user_study_round = "3"
        print('get new data, user_id:{},movie_id:{},seen_status:{},explanation_type:{},score:{},user_study_round:{}'.format(user_id,movie_id,seen_status,explanation_type,score,user_study_round))

        save_explanation_score_tosqlite(user_id,movie_id,seen_status,explanation_type,score,user_study_round)

        return 'success'
    else:
        return 'fail'


@app.route('/user_feedback')
def user_feedback():
    return render_template('user_feedback.html',title = 'Film Recommendation')

@app.route('/question_result_transfer2',methods=['GET','POST'])
def question_result_transfer2():
    if request.method == 'POST':
        user_id = request.values['user_id']
        question_result_list = request.values['question_result_list']
        if question_result_list != '':
            save_question_result2_tosqlite(user_id,question_result_list)
            return 'success'





