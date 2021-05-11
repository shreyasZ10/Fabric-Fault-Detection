from flask import Flask
from flask_bcrypt import Bcrypt 
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

app = Flask(__name__,template_folder='templates')
app.config['SECRET_KEY'] ='b1a2082e8b412c7261e17ece6b63a64f'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

app.config['UPLOAD_FOLDER'] = 'static/uploads/'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

from fabrix import routes