from flask import Flask

app = Flask(__name__,
            static_url_path='',
            static_folder='../static')
app.config['DEBUG'] = True

from app import study_routes
