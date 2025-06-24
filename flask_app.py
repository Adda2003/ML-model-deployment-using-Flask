from flask import Flask
from flask_login import LoginManager

from models import User, db

app = Flask(__name__)

app.config['DB'] = db
app.config['SECRET_KEY'] = '9OLWxND4o83j4K4iuopO'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

app.config['DB'].init_app(app)
app.config['DB'].create_all(app=app)

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    # since the user_id is just the primary key of our user table, use it in the query for the user
    return User.query.get(int(user_id))


from auth.routes import auth as auth_blueprint
from app.mnist import main as main_blueprint

from app.sentiment import sentiment as sentiment_blueprint

app.register_blueprint(auth_blueprint)
app.register_blueprint(main_blueprint)

app.register_blueprint(sentiment_blueprint)

if __name__ == '__main__':
    app.run()