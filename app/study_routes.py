from flask import render_template, request, redirect, jsonify, url_for, make_response
from uuid import uuid4
from app import app
from .model_utils import get_top_movie_ids, \
                         get_movie_name_for_id, \
                         get_movie_poster, \
                         recsys
from .db_utils import save_demographics_to_db, \
                      save_background_to_db, \
                      save_user_preferences_to_db, \
                      save_explanation_preference_to_sqlite, \
                      save_questionnaire_to_db, \
                      load_demographics_from_db
from .constants import SYSTEMS

# Template renders

@app.route('/introduction')
def introduction():
    response = make_response(render_template('study/01-introduction.html'))
    response.set_cookie('uid', str(uuid4()).split('-')[0])
    return response


@app.route('/', methods = [ 'GET' ])
def index():
    return redirect(url_for('introduction'))


@app.route('/demographics', methods = [ 'GET' ])
def demographics():
    uid = request.cookies.get('uid')
    if uid is None:
        return redirect(url_for('introduction'))
    return render_template('study/02-questionnaire-demographics.html', uid = uid)


@app.route('/background', methods = [ 'GET' ])
def background():
    uid = request.cookies.get('uid')
    if uid is None:
        return redirect(url_for('introduction'))
    return render_template('study/03-questionnaire-background.html', uid = uid)


@app.route('/cold-intro', methods = [ 'GET' ])
def cold_intro():
    uid = request.cookies.get('uid')
    if uid is None:
        return redirect(url_for('introduction'))
    return render_template('study/04-cold-start-intro.html', uid = uid)


@app.route('/cold-start', methods = [ 'GET' ])
def cold_start():
    uid = request.cookies.get('uid')
    if uid is None:
        return redirect(url_for('introduction'))
    return render_template('study/05-cold-start.html',
        uid = uid,
        threshold = 10,
        systems = list(SYSTEMS.values()))


@app.route('/system-feedback', methods = [ 'GET' ])
def system_feedback():
    uid = request.cookies.get('uid')
    if uid is None:
        return redirect(url_for('introduction'))
    return render_template('study/06-mid-study-questionnaire.html', uid = uid)


@app.route('/recommendations-intro', methods = [ 'GET' ])
def recommendations_intro():
    uid = request.cookies.get('uid')
    if uid is None:
        return redirect(url_for('introduction'))
    return render_template('study/07-recommendation-intro.html', uid = uid)


@app.route('/recommendations', methods = [ 'GET' ])
def recommendations():
    uid = request.cookies.get('uid')
    if uid is None:
        return redirect(url_for('introduction'))

    return render_template('study/08-explained-recommendations.html',
        uid = uid,
        threshold = 25,
    )


@app.route('/post-study', methods = [ 'GET' ])
def post_study_questionnaire():
    uid = request.cookies.get('uid')
    if uid is None:
        return redirect(url_for('introduction'))
    return render_template('study/09-post-study-questionnaire.html', uid = uid)


@app.route('/wrap-up', methods = [ 'GET' ])
def wrap_up():
    pass

# API Endpoints

@app.route('/api/movies/top', methods = [ 'GET' ])
def get_top_movies():
    """
    API endpoint that provides a list of the n top rated movies.
    A maximum of the top 300 can be retrieved.
    Accepts the following parameters in the query string:
    offsete: offset in the top 300 movies starting with the top movie, default 0
    count: number of items returned, default 6
    """
    offset = request.args.get('offset', default = 0, type = int)
    count = request.args.get('count', default = 6, type = int)

    ids = get_top_movie_ids(300)[offset:offset + count]

    return jsonify([ {
        'title': get_movie_name_for_id(id),
        'image': get_movie_poster(recsys.data.movies[recsys.data.movies.iid == id]),
        'id': int(id)
    } for id in ids ])


@app.route('/api/user/<uid>/recommendations/<system>', methods = [ 'GET' ])
def get_movie_recommendations_for_user(uid, system):
    """
    API endpoint that provides a set of recommendations, ignoring previously
    seen recommendations, i.e. "cold recommendations"
    """
    n = request.args.get('count') or 10
    with_explanations = False if request.args.get('explanations') is None else True
    print(with_explanations)

    if not recsys.user_is_built:
        return make_response(jsonify({ 'error': 'User not built.' }, 400))

    if system == SYSTEMS['OUR_SYSTEM']:
        recommendations, explanations = recsys.get_recommendations({'IUI': 3, 'UIU':  0, 'IUDD': 0, 'UICC': 0 })
        recommendations = list(recommendations[[ 'iid', 'title' ]].to_dict(orient='index').values())
        for rec in recommendations:
            rec['image'] = gget_movie_poster(recsys.data.movies[recsys.data.movies.iid == rec['iid']])
            rec['id']  = rec['iid']

        # TODO Add explanation if requested

        return jsonify(recommendations)
    elif system == SYSTEMS['BENCHMARK_1']:
        # TODO
        recommendations = [ ]
        return jsonify([ { 'id': 0, 'image': 'https://m.media-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@._V1_SX300.jpg', 'title': 'Toy Story 2' } ])
    elif system == SYSTEMS['BENCHMARK_2']:
        # TODO
        recommendations = [ ]
        return jsonify([ { 'id': 0, 'image': 'https://m.media-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@._V1_SX300.jpg', 'title': 'Toy Story 3' } ])
    else:
        return jsonify({ 'error': 'System %s not available' % system }, 404)
    return jsonify(recommendations)


@app.route('/api/user/<uid>/demographics', methods = [ 'POST' ])
def post_demographics_for_user(uid):
    """
    API endpoint that receives demographic information about the current user,
    stores in the DB and updates the model
    """
    demographics = request.json['demographics']
    save_demographics_to_db(uid, demographics['age'], demographics['gender'], demographics['occupation'])

    background = request.json['background']
    save_background_to_db(uid, background)

    recsys.build_user([ ], (demographics['gender'], demographics['occupation']))
    return make_response(jsonify({ 'success': True }), 202)


@app.route('/api/user/<uid>/movies/preferences', methods = [ 'POST' ])
def post_movie_preferences_for_user(uid):
    """
    API endpoint for incrementally refining the user profile.
    The endpoints takes a list of movie preferences and builds
    them into the user profile.
    """
    preference_data = request.json
    save_user_preferences_to_db(uid, preference_data)
    db_data = load_demographics_from_db(uid)
    if db_data is None:
        return make_response(jsonify({ 'error': 'User not found' }), 404)
    db_uid, age, gender, occupation = db_data
    recsys.build_user(preference_data.keys(), (gender, occupation))
    return make_response(jsonify({ 'success': True }), 202)


@app.route('/api/user/<uid>/explanations/preferences', methods = [ 'POST' ])
def post_explanation_preferences_for_user(uid):
    """
    API endpoint for posting user explanation preferences.
    """
    # TODO
    pass

@app.route('/api/user/<uid>/questionnaires/post', methods = [ 'POST' ])
def post_questionnaire(uid):
    questionnaire_data = request.json
    save_questionnaire_to_db(uid, questionnaire_data)
    return make_response(jsonify({ 'success': True }), 202)
