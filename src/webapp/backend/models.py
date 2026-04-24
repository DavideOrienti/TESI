from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    ratings       = db.relationship("Rating", backref="user", lazy=True)
    favorites     = db.relationship("Favorite", backref="user", lazy=True)


class Movie(db.Model):
    id          = db.Column(db.Integer, primary_key=True)  # = movieId MovieLens
    title       = db.Column(db.String(300), nullable=False)
    title_clean = db.Column(db.String(300))
    year        = db.Column(db.Integer)
    genres      = db.Column(db.String(500))
    overview_en = db.Column(db.Text)
    overview_it = db.Column(db.Text)
    director    = db.Column(db.String(200))
    actors_top5 = db.Column(db.String(500))
    poster_url  = db.Column(db.String(500))
    popularity  = db.Column(db.Float)
    tmdb_id     = db.Column(db.Integer)


class Rating(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    movie_id   = db.Column(db.Integer, db.ForeignKey("movie.id"), nullable=False)
    rating     = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint("user_id", "movie_id"),)


class Favorite(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    movie_id   = db.Column(db.Integer, db.ForeignKey("movie.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint("user_id", "movie_id"),)
