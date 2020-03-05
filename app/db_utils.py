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