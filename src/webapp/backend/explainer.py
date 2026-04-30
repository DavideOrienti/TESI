from flask import Blueprint, request, jsonify
import os

explainer_bp = Blueprint("explainer", __name__)

IS_RENDER = os.environ.get("RENDER", "").lower() in ("1", "true", "yes")


def _build_prompt(film: dict, seed_movies: list[dict], username: str) -> str:
    seed_titles = ", ".join([
        f"{m['title']} ({m.get('rating', ''):.1f}★)"
        for m in seed_movies[:3] if m.get('title')
    ])

    return f"""Sei un assistente di raccomandazione film.
Un utente di nome {username} ha apprezzato questi film: {seed_titles}.

Ti suggeriamo di consigliare: {film.get('title')} ({film.get('year', '')})
Generi: {film.get('genres', '').replace('|', ', ')}
Trama: {str(film.get('overview_en', ''))[:300]}
Regista: {film.get('director', '')}

Scrivi UNA spiegazione breve (massimo 2 frasi) in italiano \
del perché questo film potrebbe piacere all'utente,
citando i suoi film preferiti. Sii naturale e coinvolgente.
Rispondi SOLO con la spiegazione, senza prefissi o titoli."""


def _explain_with_ollama(prompt: str) -> str:
    import ollama
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "num_predict": 150}
    )
    return response['message']['content'].strip()


def _explain_with_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def generate_explanation(film: dict, seed_movies: list[dict],
                         username: str = "utente") -> str | None:
    """
    Genera spiegazione LLM.
    Usa Groq in produzione, Ollama in locale.
    Restituisce None se entrambi falliscono.
    """
    prompt = _build_prompt(film, seed_movies, username)

    try:
        if IS_RENDER:
            if not os.environ.get("GROQ_API_KEY"):
                print("[explainer] GROQ_API_KEY not set, skipping")
                return None
            result = _explain_with_groq(prompt)
        else:
            result = _explain_with_ollama(prompt)

        print(f"[explainer] OK — {len(result)} chars, "
              f"method={'groq' if IS_RENDER else 'ollama'}")
        return result
    except Exception as e:
        print(f"[explainer] error: {e}")
        return None


@explainer_bp.route("/explain/<int:movie_id>")
def explain_recommendation(movie_id: int):
    """
    GET /api/explain/<movie_id>
    Header: Authorization: Bearer <token> (richiesto)
    """
    from .models import Rating, Movie

    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        return jsonify({"error": "token required"}), 401

    try:
        import jwt
        from flask import current_app
        payload = jwt.decode(
            token,
            current_app.config["JWT_SECRET_KEY"],
            algorithms=["HS256"]
        )
        user_id = payload["user_id"]
        username = payload.get("username", "utente")
    except Exception:
        return jsonify({"error": "invalid token"}), 401

    film = Movie.query.get(movie_id)
    if not film:
        return jsonify({"error": "film not found"}), 404

    film_dict = {
        "title": film.title_clean or film.title,
        "year": film.year,
        "genres": film.genres or "",
        "overview_en": film.overview_en or "",
        "director": film.director or ""
    }

    top_ratings = Rating.query.filter(
        Rating.user_id == user_id,
        Rating.rating >= 4.0
    ).order_by(Rating.rating.desc()).limit(5).all()

    seed_movies = []
    for r in top_ratings:
        m = Movie.query.get(r.movie_id)
        if m:
            seed_movies.append({
                "title": m.title_clean or m.title,
                "rating": r.rating
            })

    if not seed_movies:
        return jsonify({
            "explanation": None,
            "message": "Valuta almeno un film con 4+ stelle per ricevere spiegazioni personalizzate"
        })

    explanation = generate_explanation(film_dict, seed_movies, username)

    if explanation is None:
        return jsonify({
            "explanation": None,
            "message": "Spiegazione non disponibile al momento"
        })

    return jsonify({
        "movie_id": movie_id,
        "explanation": explanation,
        "method": "groq" if IS_RENDER else "ollama",
        "seed_movies": [m["title"] for m in seed_movies[:3]]
    })


@explainer_bp.route("/explain/<int:movie_id>", methods=["OPTIONS"])
def explain_options(movie_id):
    return "", 200
