from models import User

from flask_login import (login_user, logout_user, login_required)
from flask import (Blueprint, render_template, redirect, url_for, request,
                   flash, current_app)
from werkzeug.security import (generate_password_hash, check_password_hash)

auth = Blueprint('auth', __name__)


@auth.route('/login')
def login():
    return render_template('login.html')


@auth.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    # TODO
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        flash('Invalid email or password')
        return redirect(url_for('auth.login'))

    login_user(user, remember=remember)
    return redirect(url_for('main.profile'))


@auth.route('/signup')
def signup():
    return render_template('signup.html')


@auth.route('/signup', methods=['POST'])
def signup_post():
    db = current_app.config['DB']

    email = request.form.get('email')
    name = request.form.get('name')
    password = request.form.get('password')

    # TODO

    if User.query.filter_by(email=email).first():
        flash('Email already registered')
        return redirect(url_for('auth.signup'))

    new_user = User(
        email=email,
        name=name,
        password=generate_password_hash(password, method='sha256')
    )
    db.session.add(new_user)
    db.session.commit()
    return redirect(url_for('auth.login'))


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))