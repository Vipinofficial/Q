# poll_maker_streamlit_with_tokens.py
# Enhanced Streamlit Poll Maker with:
# - Admin (password-protected: fall@table145)
# - Create polls + roster import -> generate tokens
# - Token-based participant access (single submission)
# - Per-question timers and presenter mode
# - Per-student randomized question & option order (persisted for audit)
# - Autosave draft answers via session_state
# - CSV exports for responses & tokens
# - Uses SQLite (polls.db)
#
# NOTE: This implementation avoids external JS/proctoring libs.
# For proctoring/focus-loss detection see 'streamlit-javascript' integration notes.
#
# Run:
# pip install streamlit pandas numpy
# streamlit run poll_maker_streamlit_with_tokens.py

import streamlit as st
import sqlite3, json, time, hashlib, random, string, csv, io
import pandas as pd
from collections import Counter, defaultdict

DB_PATH = "polls.db"
ADMIN_PASSWORD = "fall@table149"

# ----------------- Database -----------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS polls (
                id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                created_at INTEGER,
                config TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                poll_id TEXT,
                ts INTEGER,
                user_tag TEXT,
                segment TEXT,
                answers TEXT,
                late INTEGER DEFAULT 0
            );
        """)
        conn.commit()

# ----------------- Utilities -----------------
def now_ts(): return int(time.time())
def make_code(n=8):
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(random.choice(alphabet) for _ in range(n))

def save_poll(pid, title, desc, cfg):
    with get_conn() as conn:
        conn.execute("REPLACE INTO polls (id,title,description,created_at,config) VALUES (?,?,?,?,?)",
                     (pid, title, desc, now_ts(), json.dumps(cfg)))
        conn.commit()

def load_poll(pid):
    with get_conn() as conn:
        cur = conn.execute("SELECT id,title,description,created_at,config FROM polls WHERE id=?", (pid,))
        row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "title": row[1], "description": row[2], "created_at": row[3], "config": json.loads(row[4])}

def list_polls(limit=50):
    with get_conn() as conn:
        rows = conn.execute("SELECT id,title,description,created_at FROM polls ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [{"id": r[0], "title": r[1], "description": r[2], "created_at": r[3]} for r in rows]

def save_response(pid, user_tag, segdict, answers, late=False):
    with get_conn() as conn:
        conn.execute("INSERT INTO responses (poll_id,ts,user_tag,segment,answers,late) VALUES (?,?,?,?,?,?)",
                     (pid, now_ts(), user_tag, json.dumps(segdict or {}), json.dumps(answers or {}), 1 if late else 0))
        conn.commit()

def load_responses(pid):
    with get_conn() as conn:
        rows = conn.execute("SELECT ts,user_tag,segment,answers,late FROM responses WHERE poll_id=? ORDER BY ts ASC", (pid,)).fetchall()
    recs = []
    for r in rows:
        recs.append({"ts": r[0], "user_tag": r[1], "segment": json.loads(r[2] or "{}"), "answers": json.loads(r[3] or "{}"), "late": bool(r[4])})
    return recs

def anon_tag():
    h = st.session_state.get("_anon_tag")
    if not h:
        h = hashlib.sha1(str(time.time()).encode()).hexdigest()[:8]
        st.session_state["_anon_tag"] = h
    return h

# ----------------- Admin: Create Poll & Roster/Token Management -----------------
def admin_create_poll_ui():
    st.header("Create Poll (Admin)")
    title = st.text_input("Poll title", key="adm_title")
    desc = st.text_area("Description (optional)", key="adm_desc")

    st.info("You can optionally import questions from CSV with columns: type,prompt,options (| separated),time_seconds,points,correct")
    up = st.file_uploader("Upload question CSV (optional)", type=["csv"], key="qbank_upload")
    prefill = None
    if up:
        try:
            s = up.getvalue().decode("utf-8").splitlines()
            reader = csv.DictReader(s)
            pre = []
            for r in reader:
                q = {
                    "type": r.get("type", "multiple_choice"),
                    "prompt": r.get("prompt", ""),
                    "time_seconds": int(r.get("time_seconds") or 20),
                    "points": int(r.get("points") or 0)
                }
                opts = r.get("options", "")
                if opts:
                    if q["type"] == "image_choice":
                        parsed = []
                        for seg in opts.split("|"):
                            if "||" in seg:
                                label, url = seg.split("||", 1)
                                parsed.append({"label": label.strip(), "image": url.strip()})
                            else:
                                parsed.append(seg.strip())
                        q["options"] = parsed
                    else:
                        q["options"] = [o.strip() for o in opts.split("|") if o.strip()]
                corr = r.get("correct", "")
                if corr:
                    if q["type"] in ["multiple_select", "ranking"]:
                        q["correct"] = [c.strip() for c in corr.split("|") if c.strip()]
                    else:
                        q["correct"] = corr.strip()
                pre.append(q)
            prefill = pre
            st.success(f"Imported {len(prefill)} questions.")
        except Exception as e:
            st.error("Failed to parse CSV: " + str(e))

    n = st.number_input("How many questions?", min_value=1, max_value=100, value=len(prefill) if prefill else 3, key="adm_n")
    questions = []
    for i in range(int(n)):
        st.markdown(f"### Question {i+1}")
        p = prefill[i] if prefill and i < len(prefill) else {}
        qtype = st.selectbox("Type", ["multiple_choice", "multiple_select", "short_text", "word_cloud", "ranking", "scale", "image_choice"], key=f"qtype_{i}")
        prompt = st.text_area("Prompt", key=f"qprompt_{i}", value=p.get("prompt", ""))
        required = st.checkbox("Required", value=p.get("required", True), key=f"qreq_{i}")
        time_seconds = st.number_input("Time for this question (seconds, 0 = manual advance)", min_value=0, value=p.get("time_seconds", 20), key=f"qtime_{i}")
        points = st.number_input("Points if correct (0 = no scoring)", min_value=0, value=p.get("points", 0), key=f"qpoints_{i}")
        show_immediate = st.checkbox("Show correct answer immediately after submit?", value=p.get("show_answer_immediate", False), key=f"qshow_{i}")
        q = {"id": f"q{i+1}", "type": qtype, "prompt": prompt, "required": required,
             "time_seconds": int(time_seconds), "points": int(points), "show_answer_immediate": bool(show_immediate)}

        if qtype in ["multiple_choice", "multiple_select", "ranking", "image_choice"]:
            raw = st.text_area("Options (one per line; for image_choice use 'Label | ImageURL')", key=f"qopts_{i}", value="\n".join([ (o if isinstance(o,str) else (o.get("label","") + " | " + o.get("image",""))) for o in p.get("options", []) ]))
            opts = []
            for line in raw.splitlines():
                s = line.strip()
                if not s: continue
                if qtype == "image_choice":
                    if "|" in s:
                        lab, url = [x.strip() for x in s.split("|", 1)]
                    else:
                        lab, url = s, ""
                    opts.append({"label": lab, "image": url})
                else:
                    opts.append(s)
            q["options"] = opts
            # correct answer(s)
            if qtype in ["multiple_choice", "image_choice"]:
                correct = st.text_input("Correct option label (for scoring)", key=f"qcorrect_{i}", value=p.get("correct", ""))
                q["correct"] = correct.strip()
            elif qtype == "multiple_select":
                corr = st.text_input("Comma-separated correct options (for scoring)", key=f"qcorrect_{i}", value=",".join(p.get("correct", [])) if isinstance(p.get("correct", []), list) else p.get("correct", ""))
                q["correct"] = [c.strip() for c in corr.split(",") if c.strip()]
            elif qtype == "ranking":
                corr = st.text_input("Correct ranking (comma-separated top->bottom)", key=f"qcorrect_{i}", value=",".join(p.get("correct", [])) if p.get("correct") else "")
                q["correct"] = [c.strip() for c in corr.split(",") if c.strip()]
        elif qtype == "scale":
            mn = st.number_input("Min", value=p.get("min", 1), key=f"qmin_{i}")
            mx = st.number_input("Max", value=p.get("max", 5), key=f"qmax_{i}")
            step = st.number_input("Step", value=p.get("step", 1), key=f"qstep_{i}")
            q["min"], q["max"], q["step"] = int(mn), int(mx), int(step)
            corr = st.number_input("Correct numeric answer (optional)", value=p.get("correct", 0), key=f"qcorrect_{i}")
            q["correct"] = corr
        questions.append(q)

    st.subheader("Segmentation fields (optional, up to 3)")
    segs = []
    for i in range(3):
        nm = st.text_input(f"Segment field {i+1} name (e.g., Class, Dept)", key=f"seg_name_{i}")
        vals = st.text_input(f"Allowed values (comma separated) for {nm}", key=f"seg_vals_{i}")
        if nm:
            segs.append({"name": nm, "allowed": [v.strip() for v in vals.split(",") if v.strip()]})

    if st.button("Save Poll (Admin)"):
        pid = make_code(6)
        cfg = {"questions": questions, "segments": segs, "session": {"present": False}, "tokens": {}}
        save_poll(pid, title.strip() or f"Untitled {pid}", desc.strip(), cfg)
        st.success(f"Saved poll with code: {pid}")
        st.code(f"?mode=vote&poll={pid}", language="text")
        st.code(f"?mode=results&poll={pid}", language="text")

def admin_manage_tokens_ui(poll):
    st.subheader("Roster & Token management")
    cfg = poll["config"]
    tokens = cfg.get("tokens", {})

    uploaded = st.file_uploader("Upload roster CSV (one column: name or email) to generate tokens", type=["csv"], key="roster_up")
    if uploaded:
        try:
            txt = uploaded.getvalue().decode("utf-8").splitlines()
            rdr = csv.reader(txt)
            roster = [row[0].strip() for row in rdr if row and row[0].strip()]
            st.write(f"Loaded {len(roster)} entries")
            if st.button("Generate tokens for roster"):
                for name in roster:
                    t = make_code(8)
                    tokens[t] = {"name": name, "issued": True, "used": False, "used_ts": None, "start_ts": None, "end_ts": None, "question_snapshot": None}
                cfg["tokens"] = tokens
                save_poll(poll["id"], poll["title"], poll["description"], cfg)
                st.success("Tokens generated & saved to poll config.")
        except Exception as e:
            st.error("Failed to parse roster CSV: " + str(e))

    st.markdown("**Existing tokens (first 50)**")
    if tokens:
        df = pd.DataFrame([{"token": t, "name": d.get("name"), "used": d.get("used"), "used_ts": d.get("used_ts")} for t, d in list(tokens.items())[:200]])
        st.dataframe(df)
        if st.button("Download tokens CSV"):
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["token", "name", "issued", "used", "used_ts"])
            for t, d in tokens.items():
                w.writerow([t, d.get("name"), d.get("issued", False), d.get("used", False), d.get("used_ts")])
            st.download_button("Download tokens CSV", data=buf.getvalue().encode("utf-8"), file_name=f"tokens_{poll['id']}.csv", mime="text/csv")
    else:
        st.info("No tokens generated yet.")

# ----------------- Admin Control Panel -----------------
def admin_control_ui():
    st.header("Admin Control Panel")
    polls = list_polls(200)
    sel = st.selectbox("Pick a poll to manage", [""] + [p["id"] for p in polls], key="admin_select_poll")
    if not sel:
        st.info("Select a poll to manage or create a new one under 'Create Poll'")
        return
    poll = load_poll(sel)
    if not poll:
        st.error("Poll not found")
        return

    st.subheader(f"Managing: {poll['title']} — {poll['id']}")
    cfg = poll["config"]
    session = cfg.get("session", {"present": False})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Start Presenter Mode"):
            if cfg.get("questions"):
                # set presenter session to question 0, set advance at based on q[0].time_seconds
                first_secs = int(cfg["questions"][0].get("time_seconds", 20) or 20)
                session = {"present": True, "current_q": 0, "started_at": now_ts(), "advance_at": now_ts() + first_secs}
                cfg["session"] = session
                save_poll(sel, poll["title"], poll["description"], cfg)
                st.success("Presenter started")
    with c2:
        if st.button("Previous Q"):
            if session.get("present"):
                cur = session.get("current_q", 0)
                if cur > 0:
                    cur -= 1
                    session["current_q"] = cur
                    session["advance_at"] = now_ts() + int(cfg["questions"][cur].get("time_seconds", 20) or 20)
                    cfg["session"] = session; save_poll(sel, poll["title"], poll["description"], cfg)
                    st.success("Moved to previous question")
    with c3:
        if st.button("Next Q"):
            if session.get("present"):
                cur = session.get("current_q", 0)
                if cur + 1 < len(cfg.get("questions", [])):
                    cur += 1
                    session["current_q"] = cur
                    session["advance_at"] = now_ts() + int(cfg["questions"][cur].get("time_seconds", 20) or 20)
                    cfg["session"] = session; save_poll(sel, poll["title"], poll["description"], cfg)
                    st.success("Advanced to next question")
                else:
                    session["present"] = False; cfg["session"] = session; save_poll(sel, poll["title"], poll["description"], cfg)
                    st.info("Reached end of questions")
    with c4:
        if st.button("Stop Presenter"):
            session["present"] = False; cfg["session"] = session; save_poll(sel, poll["title"], poll["description"], cfg)
            st.success("Presenter stopped")

    st.markdown("---")
    if st.button("Reset Responses for this poll"):
        with get_conn() as conn:
            conn.execute("DELETE FROM responses WHERE poll_id=?", (sel,))
            conn.commit()
        st.success("Responses deleted")

    if st.button("Export responses CSV"):
        recs = load_responses(sel)
        if recs:
            rows = []
            for r in recs:
                base = {"ts": r["ts"], "user_tag": r["user_tag"], **{k: v for k, v in r["segment"].items()}}
                for qid, qv in r["answers"].items():
                    base[qid] = json.dumps(qv)
                base["late"] = r.get("late", False)
                rows.append(base)
            df = pd.DataFrame(rows)
            st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name=f"responses_{sel}.csv", mime="text/csv")
        else:
            st.info("No responses to export")

    st.markdown("### Tokens & roster")
    admin_manage_tokens_ui(poll)

# ----------------- Presentation logic helpers -----------------
def maybe_advance_session(poll):
    cfg = load_poll(poll["id"])["config"]
    session = cfg.get("session", {})
    if not session.get("present"):
        return cfg
    if now_ts() >= int(session.get("advance_at", 0)):
        cur = session.get("current_q", 0)
        if cur + 1 < len(cfg.get("questions", [])):
            cur += 1
            session["current_q"] = cur
            session["advance_at"] = now_ts() + int(cfg["questions"][cur].get("time_seconds", 20) or 20)
            cfg["session"] = session
            save_poll(poll["id"], poll["title"], poll["description"], cfg)
        else:
            session["present"] = False
            cfg["session"] = session
            save_poll(poll["id"], poll["title"], poll["description"], cfg)
    return cfg

# ----------------- Vote View (token workflow, randomization, autosave) -----------------
def vote_view(poll):
    st.header(poll["title"])
    if poll["description"]:
        st.write(poll["description"])

    cfg = poll["config"]
    tokens = cfg.get("tokens", {})

    # If poll has tokens, require token validation
    token_required = bool(tokens)
    token = st.session_state.get("test_token")
    validated = False

    # Read query params (use st.query_params per your request)
    qp = st.query_params
    # If instructor supplied token via URL like ?token=ABC123, prefill:
    prefill_token = (qp.get("token", [None])[0] or "").strip().upper()
    if prefill_token and "test_token" not in st.session_state:
        st.session_state["token_input_prefill"] = prefill_token

    if token_required and not st.session_state.get("test_token"):
        st.info("This test requires an access token. Ask your instructor for your token.")
        token_input = st.text_input("Enter access token", value=st.session_state.get("token_input_prefill", ""), key="token_input")
        if st.button("Validate token"):
            tok = token_input.strip().upper()
            if not tok:
                st.error("Please enter a token.")
                st.stop()
            tdata = tokens.get(tok)
            if not tdata:
                st.error("Invalid token. Make sure you typed it correctly.")
                st.stop()
            # Check server-side that token hasn't already been used (single submission)
            recs = load_responses(poll["id"])
            if any(r["user_tag"] == tok for r in recs):
                st.error("This token has already been used to submit responses.")
                st.stop()
            # Issue start_ts / end_ts and create per-token question snapshot if not present
            if not tdata.get("start_ts"):
                # compute total time as sum of per-question time_seconds >0
                total = 0
                for q in cfg.get("questions", []):
                    total += int(q.get("time_seconds", 0) or 0)
                # if total == 0, we still allow per-question manual advance; set end far in future
                tdata["start_ts"] = now_ts()
                tdata["end_ts"] = now_ts() + (total if total > 0 else 60 * 60 * 24)
            # create randomized snapshot for this token if not existing
            if not tdata.get("question_snapshot"):
                qcopy = json.loads(json.dumps(cfg.get("questions", [])))  # deep copy
                # randomize question order
                indices = list(range(len(qcopy)))
                random.shuffle(indices)
                qlist = [qcopy[i] for i in indices]
                # shuffle options for choice questions
                for q in qlist:
                    if q.get("type") in ["multiple_choice", "multiple_select", "image_choice"]:
                        opts = q.get("options", []).copy()
                        random.shuffle(opts)
                        q["options"] = opts
                tdata["question_snapshot"] = qlist
                tdata["question_order_indices"] = indices
            # persist
            tokens[tok] = tdata
            cfg["tokens"] = tokens
            save_poll(poll["id"], poll["title"], poll["description"], cfg)
            # register in session
            st.session_state["test_token"] = tok
            st.success(f"Token accepted. Good luck, {tdata.get('name', 'Student')}!")
            validated = True
    else:
        if token_required:
            validated = True  # already validated in session state
        else:
            validated = True  # token not required for this poll

    # If validated, load the question set (either entire poll or token snapshot)
    if validated:
        token = st.session_state.get("test_token")
        if token and token in cfg.get("tokens", {}):
            tdata = cfg["tokens"][token]
            # ensure token hasn't been used in another tab since validation
            recs = load_responses(poll["id"])
            if any(r["user_tag"] == token for r in recs):
                st.error("This token has already been used to submit responses (detected server-side). If this is in error, contact instructor.")
                st.stop()
            questions = tdata.get("question_snapshot", cfg.get("questions", []))
            # check time left per token
            start_ts = tdata.get("start_ts")
            end_ts = tdata.get("end_ts")
            time_left = None
            if start_ts and end_ts:
                time_left = max(0, int(end_ts - now_ts()))
            # Provide overall countdown
            if time_left is not None:
                st.metric("Time remaining for you", f"{time_left}s")
            # load drafts from session or saved snapshot
            if "draft_answers" not in st.session_state:
                st.session_state["draft_answers"] = {}
            # Render one question at a time if presenter session active, else all (but we keep token snapshot order)
            session_presenter = cfg.get("session", {}).get("present", False)
            show_questions = questions
            if cfg.get("session", {}).get("present"):
                # present single question as per presenter session current_q
                cfg_reloaded = maybe_advance_session(poll)
                cur = cfg_reloaded.get("session", {}).get("current_q", 0)
                # present the corresponding question from token snapshot: map using question_order_indices if needed
                if token and cfg.get("tokens", {}).get(token, {}).get("question_order_indices"):
                    # the token snapshot was already stored as qlist, so just use index cur within that snapshot
                    qsnap = cfg.get("tokens", {}).get(token, {}).get("question_snapshot", questions)
                    if cur < len(qsnap):
                        show_questions = [qsnap[cur]]
                    else:
                        show_questions = []
                else:
                    # fallback to poll-level question at index cur
                    if cur < len(questions):
                        show_questions = [questions[cur]]
                    else:
                        show_questions = []
                st.caption("Presenter mode — instructor is advancing questions.")
            # Render questions
            segvals = {}
            if cfg.get("segments"):
                st.subheader("Segmentation (optional)")
                for s in cfg["segments"]:
                    label = s["name"]
                    allowed = s.get("allowed", [])
                    if allowed:
                        v = st.selectbox(label, ["(skip)"] + allowed, index=0, key=f"seg_{label}")
                        segvals[label] = None if v == "(skip)" else v
                    else:
                        segvals[label] = st.text_input(label, key=f"seg_{label}")

            st.subheader("Answer the question(s)")
            answers_to_save = {}
            for q in show_questions:
                qid = q["id"]
                st.markdown(f"**{q.get('prompt')}**")
                # restore draft value if present
                draft_key = f"draft_{token}_{qid}" if token else f"draft_{qid}"
                if q.get("type") == "multiple_choice":
                    val = st.radio("Select one:", options=[o if isinstance(o, str) else o.get("label") for o in q.get("options", [])], key=draft_key)
                    st.session_state["draft_answers"][draft_key] = val
                    answers_to_save[qid] = st.session_state["draft_answers"].get(draft_key)
                elif q.get("type") == "multiple_select":
                    val = st.multiselect("Select any:", options=q.get("options", []), key=draft_key)
                    st.session_state["draft_answers"][draft_key] = val
                    answers_to_save[qid] = st.session_state["draft_answers"].get(draft_key)
                elif q.get("type") == "short_text":
                    val = st.text_input("Short answer:", value=st.session_state["draft_answers"].get(draft_key, ""), key=draft_key)
                    st.session_state["draft_answers"][draft_key] = val
                    answers_to_save[qid] = val
                elif q.get("type") == "word_cloud":
                    val = st.text_input("Type short phrase:", value=st.session_state["draft_answers"].get(draft_key, ""), key=draft_key)
                    st.session_state["draft_answers"][draft_key] = val.strip()
                    answers_to_save[qid] = val.strip()
                elif q.get("type") == "ranking":
                    rem = list(q.get("options", []))
                    ranking = []
                    for k in range(len(q.get("options", []))):
                        pick = st.selectbox(f"Pick rank {k+1}", ["(choose)"] + rem, key=f"{draft_key}_rank_{k}")
                        if pick != "(choose)" and pick in rem:
                            ranking.append(pick)
                            rem.remove(pick)
                    st.session_state["draft_answers"][draft_key] = ranking
                    answers_to_save[qid] = ranking
                elif q.get("type") == "scale":
                    val = st.slider("Choose", min_value=int(q.get("min", 1)), max_value=int(q.get("max", 5)), step=int(q.get("step", 1)), key=draft_key)
                    st.session_state["draft_answers"][draft_key] = val
                    answers_to_save[qid] = val
                elif q.get("type") == "image_choice":
                    labels = [o.get("label") for o in q.get("options", [])]
                    choice = st.radio("Select one:", labels, key=draft_key)
                    # show images
                    cols = st.columns(min(4, len(labels) if labels else 1))
                    for idx, o in enumerate(q.get("options", [])):
                        with cols[idx % max(1, len(cols))]:
                            if o.get("image"):
                                st.image(o.get("image"), caption=o.get("label"), use_column_width=True)
                            else:
                                st.write(o.get("label"))
                    st.session_state["draft_answers"][draft_key] = choice
                    answers_to_save[qid] = choice
                st.divider()

            # Submit answers (for the visible questions)
            if st.button("Submit answers"):
                # validate required fields for the visible questions
                missing = []
                for q in show_questions:
                    if not q.get("required", True):
                        continue
                    v = answers_to_save.get(q["id"])
                    if q["type"] == "ranking":
                        if not v or len(v) != len(q.get("options", [])):
                            missing.append(q["prompt"] + " (complete ranking)")
                    elif q["type"] == "multiple_select":
                        if not v:
                            missing.append(q["prompt"])
                    elif q["type"] in ["word_cloud", "short_text"]:
                        if not v:
                            missing.append(q["prompt"])
                    else:
                        if v is None or v == []:
                            missing.append(q["prompt"])
                if missing:
                    st.error("Please complete required questions:\n- " + "\n- ".join(missing))
                else:
                    # server-side single-submit check
                    token_here = token or st.session_state.get("_anon_tag") or anon_tag()
                    recs = load_responses(poll["id"])
                    if token and any(r["user_tag"] == token for r in recs):
                        st.error("This token already submitted responses (detected on server).")
                        st.stop()
                    # check token time window
                    late = False
                    if token:
                        tinfo = cfg.get("tokens", {}).get(token, {})
                        if tinfo.get("end_ts") and now_ts() > tinfo.get("end_ts"):
                            late = True
                    # prepare full answers payload using drafts for all questions or visible set
                    # In presenter mode we only saved visible Qs; that's fine. In full-test mode you may want to save full set.
                    segvals["_token"] = token
                    # merge answers for visible q's from drafts
                    answers_payload = {}
                    for q in show_questions:
                        draft_key = f"draft_{token}_{q['id']}" if token else f"draft_{q['id']}"
                        answers_payload[q["id"]] = st.session_state["draft_answers"].get(draft_key)
                    # save response
                    user_tag = token if token else anon_tag()
                    save_response(poll["id"], user_tag, segvals, answers_payload, late=late)
                    # mark token used (if token exists)
                    if token:
                        cfg = load_poll(poll["id"])["config"]
                        cfg["tokens"][token]["used"] = True
                        cfg["tokens"][token]["used_ts"] = now_ts()
                        save_poll(poll["id"], poll["title"], poll["description"], cfg)
                    st.success("Response recorded! Thank you.")
                    if any(q.get("show_answer_immediate") and q.get("correct") for q in show_questions):
                        for q in show_questions:
                            if q.get("show_answer_immediate") and q.get("correct") is not None:
                                st.info(f"Correct answer for: {q['prompt']} -> {q.get('correct')}")
                    st.balloons()
                    # optionally clear drafts for submitted q's
                    for q in show_questions:
                        dk = f"draft_{token}_{q['id']}" if token else f"draft_{q['id']}"
                        if dk in st.session_state.get("draft_answers", {}):
                            del st.session_state["draft_answers"][dk]
                    st.experimental_rerun()

# ----------------- Results UI (aggregations & leaderboard) -----------------
def comput
