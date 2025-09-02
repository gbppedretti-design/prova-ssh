import os, sys, json, argparse, time
import psycopg2
from psycopg2.extras import RealDictCursor

def query_db(conn, q, topk, db_timeout):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # 1) Prova FTS se esiste search_fts
        try:
            cur.execute("""
                SELECT id, name, aliases, tags, notes
                FROM bio_items
                WHERE search_fts @@ plainto_tsquery('simple', %s)
                ORDER BY updated_at DESC
                LIMIT %s
            """, (q, topk))
            rows = cur.fetchall()
            return rows
        except Exception:
            conn.rollback()
        # 2) Fallback: match testuale su colonne "concatenate" (se presenti) o su campi base
        try:
            cur.execute("""
                SELECT id, name, aliases, tags, notes
                FROM bio_items
                WHERE
                    COALESCE(name,'') ILIKE %(p)s
                    OR COALESCE(notes,'') ILIKE %(p)s
                    OR COALESCE(tags_concatenated,'') ILIKE %(p)s
                    OR COALESCE(notes_concatenated,'') ILIKE %(p)s
                ORDER BY id ASC
                LIMIT %(k)s
            """, {"p": f"%{q}%", "k": topk})
            rows = cur.fetchall()
            return rows
        except Exception as e:
            conn.rollback()
            raise e

def maybe_llm_answer(texts, lang, llm_timeout):
    # Attiva LLM solo se OPENAI_API_KEY presente e non --no-llm
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not texts:
        return None
    try:
        # Lazy import per non richiedere dipendenza se non serve
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            f"Language: {lang}\n"
            f"Contesto (lista voci):\n- " + "\n- ".join(texts) +
            "\n\nRispondi in modo sintetico."
        )
        # Timeout “soft”: interrompiamo se supera llm_timeout
        start = time.time()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        if time.time() - start > llm_timeout:
            return None
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Query testuale")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--lang", default="italian", choices=["italian","english"])
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--db-timeout", type=int, default=5, help="Timeout connessione DB (s)")
    ap.add_argument("--llm-timeout", type=int, default=15, help="Timeout risposta LLM (s)")
    args = ap.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print(json.dumps({"error": "DATABASE_URL mancante"}))
        sys.exit(1)

    try:
        conn = psycopg2.connect(db_url, connect_timeout=args.db_timeout)
    except Exception as e:
        print(json.dumps({"error": f"Connessione DB fallita: {str(e)}"}))
        sys.exit(2)

    try:
        rows = query_db(conn, args.q, args.topk, args.db_timeout)
    finally:
        conn.close()

    results = []
    for r in rows:
        results.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "aliases": r.get("aliases"),
            "tags": r.get("tags"),
            "notes": r.get("notes"),
        })

    payload = {
        "query": args.q,
        "lang": args.lang,
        "count": len(results),
        "results": results
    }

    # LLM opzionale
    if not args.no_llm:
        texts = [f"{x.get('name','')}: {x.get('notes','')}" for x in results]
        llm_ans = maybe_llm_answer(texts, args.lang, args.llm_timeout)
        if llm_ans:
            payload["llm_answer"] = llm_ans

    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
