import os.path
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "MATDemo.db")


demographics_table = 'demographics'
create_demographics_table = 'CREATE TABLE IF NOT EXISTS `%s` (\
    user_id VARCHAR NOT NULL, \
    age INT NUT NULL, \
    gender CHAR NOT NULL, \
    occupation INT NOT NULL\
)' % demographics_table

background_table = 'background'
create_background_table = 'CREATE TABLE IF NOT EXISTS `%s` (\
    user_id VARCHAR NOT NULL,\
    watch_frequency INT NOT NULL,\
    recommendation_frequency INT NOT NULL,\
    recommendation_system_frequency INT NOT NULL,\
    recommendation_familiarity INT NOT NULL,\
    explainable_recommendation_familiarity INT NOT NULL,\
    recommender_experience INT NOT NULL,\
    system_confidence INT NOT NULL\
)' % background_table

preference_table = 'movie_preferences'
create_preference_table = 'CREATE TABLE IF NOT EXISTS `%s` (\
    user_id VARCHAR NOT NULL,\
    movie_id INT NOT NULL,\
    is_preferred BOOLEAN NOT NULL DEFAULT FALSE\
)' % preference_table


explanation_preference_table = 'explanations'
create_explanation_preference_table = 'CREATE TABLE IF NOT EXISTS `%s` (\
    user_id VARCHAR NOT NULL,\
    movie_id INT NOT NULL,\
    expl_type VARCHAR NOT NULL,\
    expl_text TEXT NOT NULL DEFAULT \'\',\
    vote INT NOT NULL\
)' % explanation_preference_table

questionnaire_data_table = 'questionnaire_data'
create_questionnaire_data_table = 'CREATE TABLE IF NOT EXISTS `%s` (\
    user_id VARCHAR NOT NULL,\
    questionnaire_data TEXT\
)' % questionnaire_data_table


connection = sqlite3.connect(db_path)
cursor = connection.cursor()
cursor.execute(create_demographics_table)
cursor.execute(create_background_table)
cursor.execute(create_preference_table)
cursor.execute(create_explanation_preference_table)
cursor.execute(create_questionnaire_data_table)
connection.commit()
connection.close()


def save_demographics_to_db(user_id, age, gender, occupation):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    params = (
        user_id,
        age,
        gender,
        occupation
    )

    cursor.execute('INSERT INTO `%s` VALUES (?, ?, ?, ?)' % demographics_table, params)
    connection.commit()
    connection.close()
    return True


def load_demographics_from_db(user_id):
    print(user_id)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM `%s` WHERE user_id = ?' % demographics_table, (user_id, ))
    demographics = cursor.fetchone()
    connection.commit()
    connection.close()
    return demographics


def save_background_to_db(user_id, background):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    params = (
        user_id,
        background['watch_frequency'],
        background['recommendation_frequency'],
        background['recommendation_system_frequency'],
        background['recommendation_familiarity'],
        background['explainable_recommendation_familiarity'],
        background['recommender_experience'],
        background['system_confidence'],
    )

    cursor.execute('INSERT INTO `%s` VALUES (?, ?, ?, ?, ?, ?, ?, ?)' % background_table, params)
    connection.commit()
    connection.close()
    return True


def save_user_preferences_to_db(user_id, preference_data):
    """
    Stores user preferences regarding recommended movies in the database.
    Expects the user id and a dictionary movie_id -> is_preferred, e.g. { 4123: True, 4102: False }
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    for movie_id in preference_data:
        params = (user_id, movie_id, preference_data[movie_id])
        cursor.execute('INSERT INTO `%s` VALUES (?, ?, ?)' % preference_table, params)

    connection.commit()
    connection.close()
    return True


def save_explanation_preference_to_sqlite(user_id, explanation_preferences):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    for movie_id, expl_type, expl_text, vote in explanation_preferences:
        params = (user_id, movie_id, expl_type, expl_text or '', vote)
        cursor.execute('INSERT INTO `%s` VALUES (?, ?, ?, ?, ?)' % explanation_preference_table, params)

    connection.commit()
    connection.close()
    return True


def save_questionnaire_to_db(user_id, questionnaire_data):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute('INSERT INTO `%s` VALUES (?, ?)' % questionnaire_data_table, (user_id, questionnaire_data_table))

    connection.commit()
    connection.close()
    return True
