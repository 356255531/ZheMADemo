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


connection = sqlite3.connect(db_path)
cursor = connection.cursor()
cursor.execute(create_demographics_table)
cursor.execute(create_background_table)
# TODO create all other tables once on startup!
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
    # TODO
    pass


def save_questionnaire_to_db(user_id, questionnaire_data):
    # TODO
    pass


def save_explanation_score_to_sqlite(user_id,movie_id,seen_status,explanation_type,explanation_score,user_study_round):
    connection = sqlite3.connect(db_path)

    cursor = connection.cursor()
    print("Opened database successfully")

    cursor.execute('CREATE TABLE IF NOT EXISTS EXP_SCORE (user_id,movie_id,seen_status,explanation_type,explanation_score,user_study_round)')
    params = (user_id,movie_id,seen_status,explanation_type,explanation_score,user_study_round)

    cursor.execute("INSERT INTO EXP_SCORE VALUES (?,?,?,?,?,?)",params)
    connection.commit()
    print("Records created successfully")

    connection.close()
    return 1


def save_question_result1_tosqlite(user_id,question_result1_list):
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


def save_question_result2_tosqlite(user_id,question_result2_list):
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


def select_explanation_type_and_score(user_id,round_number):
    connection = sqlite3.connect(db_path)

    cursor = connection.cursor()
    print("Opened database successfully")

    params = (str(user_id),str(round_number))
    cursor.execute("SELECT explanation_type, explanation_score FROM EXP_SCORE WHERE user_id=? AND user_study_round=?",params)

    explanation_type_score_rows = cursor.fetchall()

    connection.commit()
    print("Select data successfully")

    return explanation_type_score_rows;
