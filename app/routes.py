from flask import render_template, request
from app import app


# @app.route('/')
# @app.route('/index')
def index():
    user = {'username':'Zhe'}
    return render_template('explanation.html', title = 'Film Recommendation', user = user)

# @app.route('/explanation')
def explanation():
    return render_template('explanation.html', title = 'Film Recommendation')

# @app.route('/user_information')
def user_information():
    return render_template('user_information.html', title = 'Film Recommendation')

# @app.route('/user_background')
def user_background():
    return render_template('user_background.html',title = 'Film Recommendation')

# @app.route('/question_result_transfer',methods=['GET','POST'])
def question_result_transfer():
    if request.method == 'POST':
        user_id = request.values['user_id']
        global current_user_id
        current_user_id = user_id
        question_result_list = request.values['question_result_list']
        if question_result_list != '':
            save_question_result1_tosqlite(user_id,question_result_list)
            return 'success'

# @app.route('/movie_preview')
def movie_preview():

    movie_ids = generateIDs(500)
    step = 6
    group_movieIDs = [movie_ids[i:i + step] for i in range(0, len(movie_ids), step)]
    click_count = refresh_value
    # import pdb
    # pdb.set_trace()

    return render_template('movie_preview.html',title = 'Film Recommendation', group_movieIDs = group_movieIDs, click_count = click_count)

# @app.route('/refresh_count',methods=['GET','POST'])
def refresh_count():
    if request.method == 'POST':
        temp_refresh_value = request.values['refresh_value']
        global refresh_value
        if temp_refresh_value != '':
            refresh_value = int(temp_refresh_value)
            return 'success'

# @app.route('/imgID_userinfo_transfer',methods=['GET','POST'])
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


# @app.route('/movie_degree')
def movie_degree():

    global iid_list
    global demographic_info
    global rs_proportion

    recsys.build_user(iid_list, demographic_info)
    print('new user created')

    df, exps = recsys.get_recommendations(rs_proportion)
    rec_movie_iids = df.iid.values

    return render_template('movie_degree.html',title = 'Film Recommendation',rec_movie_iids_and_explanations = zip(rec_movie_iids,exps))

# @app.route('/movie_name_transfer',methods=['GET','POST'])
def movie_name_transfer():
    if request.method == 'POST':
        movie_id = request.values['movie_id']
        movie_name = get_movie_name_withID(movie_id)
        return movie_name

# @app.route('/score_movie_transfer',methods=['GET','POST'])
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

# @app.route('/user_info_transfer',methods=['GET','POST'])
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

# @app.route('/movie_degree2')
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

# @app.route('/score_movie_transfer2',methods=['GET','POST'])
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

# @app.route('/movie_degree3')
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

# @app.route('/score_movie_transfer3',methods=['GET','POST'])
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


# @app.route('/user_feedback')
def user_feedback():
    return render_template('user_feedback.html',title = 'Film Recommendation')

# @app.route('/question_result_transfer2',methods=['GET','POST'])
def question_result_transfer2():
    if request.method == 'POST':
        user_id = request.values['user_id']
        question_result_list = request.values['question_result_list']
        if question_result_list != '':
            save_question_result2_tosqlite(user_id,question_result_list)
            return 'success'
