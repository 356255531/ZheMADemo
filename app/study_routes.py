from flask import render_template, request, redirect, jsonify, url_for, make_response
from uuid import uuid4
from app import app
from .model_utils import get_top_movie_ids, get_movie_name_for_id, get_movie_poster_for_id, recsys
from .db_utils import save_demographics_to_db, save_background_to_db

# Template renders

@app.route('/introduction')
def introduction():
    response = make_response(render_template('study/01-introduction.html'))
    response.set_cookie('uid', str(uuid4()))
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
        get_movies_url = url_for('get_movie_recommendations_for_user_cold', uid = uid),
        next = 'incomplete_cold_start')


@app.route('/incomplete-cold-start', methods = [ 'GET' ])
def incomplete_cold_start():
    uid = request.cookies.get('uid')
    if uid is None:
        return redirect(url_for('introduction'))
    return render_template('study/05-cold-start.html',
        uid = uid,
        get_movies_url = url_for('get_movie_recommendations_for_user_incomplete_cold', uid = uid),
        next = 'system_feedback')


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
    return render_template('study/08-explained-recommendations.html', uid = uid)


@app.route('/post-study', methods = [ 'GET' ])
def post_study_questionnaire():
    uid = request.cookies.get('uid')
    if uid is None:
        return redirect(url_for('introduction'))
    return render_template('study/09-post-study-questionnaire.html', uid = uid)


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

    ids = get_ids(300)[offset:offset + count]

    return jsonify([ {
        'title': get_movie_name_for_id(id),
        'image': get_movie_poster_for_id(id),
        'id': int(id)
    } for id in ids ])


@app.route('/api/user/<uid>/movies/recommendations/cold', methods = [ 'GET' ])
def get_movie_recommendations_for_user_cold(uid):
    """
    API endpoint that provides a set of recommendations, ignoring previously
    seen recommendations, i.e. "cold recommendations"
    """
    # TODO Determine whether to use our or the reference method for this user
    # TODO Retrieve recommendations from the model
    df, explanations = recsys.get_recommendations([ ])
    # TODO Send recommendations as JSON to client
    return make_response(jsonify({ 'error': 'Not Implemented'}), 501)


@app.route('/api/user/<uid>/movies/recommendations/incomplete-cold', methods = [ 'GET' ])
def get_movie_recommendations_for_user_incomplete_cold(uid):
    """
    API endpoint that provides a set of recommendations, including previously
    seen recommendations, i.e. "incomplete cold recommendations"
    """
    # TODO Retrieve recommendations from the model
    # TODO Send recommendations as JSON to client
    return make_response(jsonify({ 'error': 'Not Implemented'}), 501)



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
    # TODO
    return make_response(jsonify({ 'error': 'Not Implemented'}), 501)


@app.route('/api/questionnaires/post', methods = [ 'POST' ])
def post_questionnaire():
    return make_response(jsonify({ 'error': 'Not Implemented'}), 501)
