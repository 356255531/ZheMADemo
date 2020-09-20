from flask import render_template, request, redirect, jsonify, url_for, make_response
from uuid import uuid4
from app import app
from .utils import get_movie_name_for_id, \
                         get_movie_poster
from .db_utils import save_demographics_to_db, \
                      save_background_to_db, \
                      save_user_preferences_to_db, \
                      save_explanation_preference_to_sqlite, \
                      save_questionnaire_to_db, \
                      load_demographics_from_db

from .ecfkg import ECFKGRecsys

SYSTEM = 'ECFKGRecsys'
SYSTEMS = {
  'OUR_SYSTEM': 'peagat',
  'ECFKGRecsys': ECFKGRecsys,
  'BENCHMARK_2': 'benchmark-2'
}
recsys = SYSTEMS[SYSTEM]()

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
    Unused

    API endpoint that provides a list of the n top rated movies.
    A maximum of the top 300 can be retrieved.
    Accepts the following parameters in the query string:
    :param offset: offset in the top 300 movies starting with the top movie, default 0
    :param count: number of items returned, default 6
    """
    offset = request.args.get('offset', default = 0, type = int)
    count = request.args.get('count', default = 6, type = int)

    ids = recsys.get_top_n_popular_items(300).iid[offset:offset + count]

    return jsonify([ {
        'title': get_movie_name_for_id(id),
        'image': get_movie_poster(recsys.data.movies[recsys.data.movies.iid == id]),
        'id': int(id)
    } for id in ids ])


@app.route('/api/user/<uid>/recommendations/<system>', methods = [ 'GET' ])
def get_movie_recommendations_for_user(uid, system):
    """
    API endpoint that provides a set of recommendations
    """
    n = request.args.get('count', default = 10, type = int)
    with_explanations = request.args.get('explanations', default = False, type = bool)

    if not recsys.user_is_built:
        return make_response(jsonify({ 'error': 'User not built.' }), 400)

    if SYSTEM == 'PEAGAT' or SYSTEM == 'ECFKGRecsys':
        recommendations, explanations, explanation_types = recsys.get_recommendations(
            10 # TODO
        )
        recommendations = list(recommendations[[ 'iid', 'title', 'year' ]].to_dict(orient='index').values())
        for rec, expl, expl_type in zip(recommendations, explanations, explanation_types):
            rec['image'] = get_movie_poster(rec)
            rec['id']  = rec['iid']
            if with_explanations:
                rec['explanation'] = { 'type': expl_type, 'text': expl }

        return jsonify(recommendations)

    elif SYSTEM == SYSTEMS['BENCHMARK_2']:
        # TODO access other recsys
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

    recsys.build_user([], (demographics['age'], demographics['gender'], demographics['occupation']))
    return make_response(jsonify({ 'success': True }), 202)


@app.route('/api/user/<uid>/movies/preferences', methods = [ 'POST' ])
def post_movie_preferences_for_user(uid):
    """
    API endpoint for incrementally refining the user profile.
    The endpoints takes a list of movie preferences and builds
    them into the user profile.
    """
    preference_data = request.json
    save_user_preferences_to_db(uid, preference_data, True)
    db_data = load_demographics_from_db(uid)
    if db_data is None:
        return make_response(jsonify({ 'error': 'User not found' }), 404)
    db_uid, age, gender, occupation = db_data
    recsys.build_user([int(key) for key, is_preferred in preference_data.items() if is_preferred], (age, gender, '%i' % occupation))
    return make_response(jsonify({ 'success': True }), 202)


@app.route('/api/user/<uid>/explanations/preferences', methods = [ 'POST' ])
def post_explanation_preferences_for_user(uid):
    """
    API endpoint for posting user explanation preferences.
    """
    # TODO transform the request.json body to db compatible format
    explanation_preferences = [ ]
    movie_preferences = [ ]

    save_user_preferences_to_db(uid, movie_preferences, False)
    save_explanation_preference_to_sqlite(uid, explanation_preferences)

    db_data = load_demographics_from_db(uid)
    if db_data is None:
        return make_response(jsonify({ 'error': 'User not found' }), 404)
    db_uid, age, gender, occupation = db_data

    recsys.build_cold_user([int(key) for key, is_preferred in movie_preferences.items() if is_preferred], (age, gender, '%i' % occupation))
    return make_response(jsonify({'success': True}), 202)

@app.route('/api/user/<uid>/questionnaires/post', methods = [ 'POST' ])
def post_questionnaire(uid):
    questionnaire_data = request.json
    save_questionnaire_to_db(uid, questionnaire_data)
    return make_response(jsonify({ 'success': True }), 202)
