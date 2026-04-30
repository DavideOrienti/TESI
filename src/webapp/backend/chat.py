from flask import Blueprint, request, jsonify
import os
import json

chat_bp = Blueprint("chat", __name__)

IS_RENDER = os.environ.get("RENDER", "").lower() in ("1", "true", "yes")

SYSTEM_PROMPT = """Sei CineBot, un assistente esperto di cinema integrato \
nella piattaforma CineRec. Parli sempre in italiano.

Hai accesso al profilo dell'utente e puoi:
1. Consigliare film personalizzati basandoti sui suoi gusti
2. Rispondere a domande su film, registi, attori
3. Cercare film per trama, genere, periodo, mood
4. Conversare liberamente di cinema

Quando consigli film, usa questo formato JSON nel tuo messaggio:
<films>{{"movies": [{{"title": "...", "year": ..., "reason": "..."}}]}}</films>

Sii amichevole, entusiasta del cinema e conciso (max 3-4 frasi per risposta).
Non inventare film che non esistono.

Profilo utente:
{user_profile}
"""


def _get_groq_response(messages: list, system: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system}] + messages,
        max_tokens=500,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()


def _get_ollama_response(messages: list, system: str) -> str:
    import ollama
    full_messages = [{"role": "system", "content": system}] + messages
    response = ollama.chat(
        model="llama3.2",
        messages=full_messages,
        options={"temperature": 0.8, "num_predict": 500}
    )
    return response['message']['content'].strip()


def _enrich_with_search(user_message: str, top_k: int = 3) -> str:
    try:
        from .search import _tfidf_vectorizer, _tfidf_matrix, _tfidf_index, _tfidf_ready
        if not _tfidf_ready:
            return ""

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = _tfidf_vectorizer.transform([user_message])
        sims = cosine_similarity(query_vec, _tfidf_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if sims[idx] < 0.05:
                break
            from .models import Movie
            mid = int(_tfidf_index.iloc[idx]["movieId"])
            m = Movie.query.get(mid)
            if m:
                results.append(
                    f"- {m.title_clean or m.title} ({m.year}): "
                    f"{str(m.overview_en or '')[:150]}"
                )

        if results:
            return "\n\nFilm rilevanti nel catalogo:\n" + "\n".join(results)
        return ""
    except Exception as e:
        print(f"[chat] search error: {e}")
        return ""


def _build_user_profile(user_id: int) -> str:
    try:
        from .models import Rating, Movie, Favorite

        top_ratings = Rating.query.filter(
            Rating.user_id == user_id,
            Rating.rating >= 4.0
        ).order_by(Rating.rating.desc()).limit(5).all()

        rated_titles = []
        for r in top_ratings:
            m = Movie.query.get(r.movie_id)
            if m:
                rated_titles.append(
                    f"{m.title_clean or m.title} ({r.rating:.1f}★)"
                )

        favorites = Favorite.query.filter_by(user_id=user_id).limit(3).all()

        fav_titles = []
        for f in favorites:
            m = Movie.query.get(f.movie_id)
            if m:
                fav_titles.append(m.title_clean or m.title)

        profile_parts = []
        if rated_titles:
            profile_parts.append(f"Film apprezzati: {', '.join(rated_titles)}")
        if fav_titles:
            profile_parts.append(f"Preferiti: {', '.join(fav_titles)}")
        if not profile_parts:
            return "Utente nuovo, nessuna preferenza registrata ancora."

        return "\n".join(profile_parts)
    except Exception as e:
        return "Profilo non disponibile."


@chat_bp.route("/message", methods=["POST"])
def chat_message():
    """
    POST /api/chat/message
    Body: {"message": str, "history": [{"role": "user"|"assistant", "content": str}]}
    Auth: opzionale (JWT)
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "message required"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "empty message"}), 400

    history = data.get("history", [])[-10:]

    user_profile = "Utente non autenticato."
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token:
        try:
            import jwt
            from flask import current_app
            payload = jwt.decode(
                token,
                current_app.config["JWT_SECRET_KEY"],
                algorithms=["HS256"]
            )
            user_id = payload.get("user_id")
            if user_id:
                user_profile = _build_user_profile(user_id)
        except Exception:
            pass

    catalog_context = _enrich_with_search(user_message)
    system = SYSTEM_PROMPT.format(user_profile=user_profile) + catalog_context
    messages = list(history) + [{"role": "user", "content": user_message}]

    try:
        if IS_RENDER:
            if not os.environ.get("GROQ_API_KEY"):
                return jsonify({"error": "LLM not configured"}), 503
            response_text = _get_groq_response(messages, system)
        else:
            response_text = _get_ollama_response(messages, system)
    except Exception as e:
        print(f"[chat] LLM error: {e}")
        return jsonify({"error": f"LLM error: {str(e)}"}), 500

    suggested_movies = []
    # Prova 1: tag <films>...</films>
    if "<films>" in response_text and "</films>" in response_text:
        try:
            start = response_text.index("<films>") + 7
            end = response_text.index("</films>")
            films_data = json.loads(response_text[start:end])
            suggested_movies = films_data.get("movies", [])
            response_text = (
                response_text[:response_text.index("<films>")].strip()
                + " " + response_text[end + 8:].strip()
            ).strip()
        except Exception:
            pass
    # Prova 2: JSON grezzo {"movies": [...]} nel testo (llm piccoli usano code block)
    if not suggested_movies:
        import re
        match = re.search(r'\{[\s\S]*?"movies"\s*:\s*\[[\s\S]*?\]\s*\}', response_text)
        if match:
            try:
                films_data = json.loads(match.group())
                suggested_movies = films_data.get("movies", [])
                response_text = response_text[:match.start()].strip() + response_text[match.end():].strip()
                # Rimuovi eventuali backtick residui
                response_text = re.sub(r'```[a-z]*\n?', '', response_text).strip()
            except Exception:
                pass

    enriched_movies = []
    for film in suggested_movies[:5]:
        try:
            from .models import Movie
            m = Movie.query.filter(
                Movie.title_clean.ilike(f"%{film['title']}%")
            ).first()
            if m:
                enriched_movies.append({
                    "movie_id": m.id,
                    "title": m.title_clean or m.title,
                    "year": m.year,
                    "poster_url": m.poster_url,
                    "genres": m.genres,
                    "reason": film.get("reason", "")
                })
        except Exception:
            pass

    print(f"[chat] OK — method={'groq' if IS_RENDER else 'ollama'}, "
          f"films={len(enriched_movies)}, chars={len(response_text)}")

    return jsonify({
        "response": response_text,
        "suggested_movies": enriched_movies,
        "method": "groq" if IS_RENDER else "ollama"
    })


@chat_bp.route("/message", methods=["OPTIONS"])
def chat_options():
    return "", 200
