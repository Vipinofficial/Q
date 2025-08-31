# poll_maker_streamlit_full.py
# Enhanced Streamlit Poll Maker
# Features:
# - Admin (password-protected: fall@table145)
# - Create polls with per-question time, points, and correct answers
# - Presenter mode: auto-advance, big countdown, progress bar, 3s warning
# - Leaderboard (if questions have correct answers and points)
# - Export responses CSV; import question bank CSV (simple format)
# - Mobile-friendly large buttons, quick submit, short-text, word-cloud, multi choice, multi-select
# - Uses SQLite for local persistence (polls.db)
#
# Run:
# pip install streamlit pandas numpy
# streamlit run poll_maker_streamlit_full.py

import streamlit as st
import pandas as pd, numpy as np, sqlite3, json, time, hashlib, random, string, io, csv
from collections import Counter, defaultdict

DB_PATH = "polls.db"
ADMIN_PASSWORD = "fall@table145"

# ----------------- DB -----------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS polls (
            id TEXT PRIMARY KEY, title TEXT, description TEXT, created_at INTEGER, config TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT, poll_id TEXT, ts INTEGER, user_tag TEXT, segment TEXT, answers TEXT)""")
        conn.commit()

# ----------------- Utilities -----------------
def now_ts(): return int(time.time())
def make_code(n=6):
    letters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(letters) for _ in range(n))

def save_poll(pid, title, desc, cfg):
    with get_conn() as conn:
        conn.execute("REPLACE INTO polls (id,title,description,created_at,config) VALUES (?,?,?,?,?)",
                     (pid, title, desc, now_ts(), json.dumps(cfg)))
        conn.commit()

def load_poll(pid):
    with get_conn() as conn:
        cur = conn.execute("SELECT id,title,description,created_at,config FROM polls WHERE id=?", (pid,))
        row = cur.fetchone()
    if not row: return None
    return {"id":row[0],"title":row[1],"description":row[2],"created_at":row[3],"config":json.loads(row[4])}

def list_polls(limit=50):
    with get_conn() as conn:
        rows = conn.execute("SELECT id,title,description,created_at FROM polls ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [{"id":r[0],"title":r[1],"description":r[2],"created_at":r[3]} for r in rows]

def save_response(pid, user_tag, segdict, answers):
    with get_conn() as conn:
        conn.execute("INSERT INTO responses (poll_id,ts,user_tag,segment,answers) VALUES (?,?,?,?,?)",
                     (pid, now_ts(), user_tag, json.dumps(segdict or {}), json.dumps(answers or {})))
        conn.commit()

def load_responses(pid):
    with get_conn() as conn:
        rows = conn.execute("SELECT ts,user_tag,segment,answers FROM responses WHERE poll_id=? ORDER BY ts ASC", (pid,)).fetchall()
    recs = []
    for r in rows:
        recs.append({"ts":r[0],"user_tag":r[1],"segment": json.loads(r[2] or "{}"), "answers": json.loads(r[3] or "{}")})
    return recs

def anon_tag():
    h = st.session_state.get("_anon_tag")
    if not h:
        h = hashlib.sha1(str(time.time()).encode()).hexdigest()[:8]
        st.session_state["_anon_tag"] = h
    return h

# ----------------- UI helpers -----------------
def q_block_admin(i, prefill=None):
    st.markdown(f"### Question {i+1}")
    p = prefill or {}
    qtype = st.selectbox("Type", ["multiple_choice","multiple_select","short_text","word_cloud","ranking","scale","image_choice"], key=f"qtype_{i}")
    prompt = st.text_area("Prompt", key=f"qprompt_{i}", value=p.get("prompt",""))
    required = st.checkbox("Required", value=p.get("required", True), key=f"qreq_{i}")
    time_seconds = st.number_input("Time for this question (seconds, 0 = manual advance)", min_value=0, value=p.get("time_seconds", 20), key=f"qtime_{i}")
    points = st.number_input("Points for correct answer (0 = no scoring)", min_value=0, value=p.get("points", 0), key=f"qpoints_{i}")
    show_answer_immediate = st.checkbox("Show correct answer immediately after submit?", value=p.get("show_answer_immediate", False), key=f"qshow_{i}")
    q = {"id":f"q{i+1}","type":qtype,"prompt":prompt,"required":required,"time_seconds":int(time_seconds),"points":int(points),"show_answer_immediate":bool(show_answer_immediate)}
    if qtype in ["multiple_choice","multiple_select","ranking","image_choice"]:
        raw = st.text_area("Options (one per line; for image_choice use 'Label | ImageURL')", key=f"qopts_{i}", value="\n".join([o if isinstance(o,str) else (o.get('label','')+' | '+o.get('image','')) for o in p.get('options',[])]))
        opts = []
        for line in raw.splitlines():
            s=line.strip()
            if not s: continue
            if qtype == "image_choice":
                if "|" in s:
                    lab,url = [x.strip() for x in s.split("|",1)]
                else:
                    lab,url = s,""
                opts.append({"label":lab,"image":url})
            else:
                opts.append(s)
        q["options"]=opts
        # correct answer input for scoring
        if qtype in ["multiple_choice","image_choice"]:
            correct = st.text_input("Correct option label (for scoring)", key=f"qcorrect_{i}", value=p.get("correct",""))
            q["correct"] = correct.strip()
        elif qtype == "multiple_select":
            corr = st.text_input("Comma-separated correct options (for scoring)", key=f"qcorrect_{i}", value=",".join(p.get("correct",[])) if isinstance(p.get("correct",[]),list) else p.get("correct",""))
            q["correct"] = [c.strip() for c in corr.split(",") if c.strip()]
        elif qtype == "ranking":
            # store correct ranking as comma-separated
            corr = st.text_input("Correct ranking (comma-separated top->bottom)", key=f"qcorrect_{i}", value=",".join(p.get("correct",[])) if p.get("correct") else "")
            q["correct"] = [c.strip() for c in corr.split(",") if c.strip()]
    elif qtype == "scale":
        mn = st.number_input("Min", value=p.get("min",1), key=f"qmin_{i}")
        mx = st.number_input("Max", value=p.get("max",5), key=f"qmax_{i}")
        step = st.number_input("Step", value=p.get("step",1), key=f"qstep_{i}")
        q["min"],q["max"],q["step"]=int(mn),int(mx),int(step)
        corr = st.number_input("Correct numeric answer (optional)", value=p.get("correct",0), key=f"qcorrect_{i}")
        q["correct"] = corr
    return q

def create_poll_admin_ui():
    st.header("Create / Edit Poll (Admin)")
    title = st.text_input("Poll title", key="adm_title")
    desc = st.text_area("Description (optional)", key="adm_desc")
    # quick CSV import for questions
    st.info("You may import questions from a CSV with columns: type,prompt,options (pipe-separated),time_seconds,points,correct")
    up = st.file_uploader("Upload question CSV (optional)", type=["csv"])
    prefill_questions = None
    if up:
        try:
            txt = up.read().decode("utf-8").splitlines()
            reader = csv.DictReader(txt)
            pre = []
            for r in reader:
                q = {"type": r.get("type","multiple_choice"), "prompt": r.get("prompt",""), "time_seconds": int(r.get("time_seconds",20) or 20), "points": int(r.get("points",0) or 0)}
                opts = r.get("options","")
                if opts:
                    q["options"] = [o.strip() for o in opts.split("|") if o.strip()]
                corr = r.get("correct","")
                if corr:
                    if q["type"] in ["multiple_select","ranking"]:
                        q["correct"] = [c.strip() for c in corr.split("|") if c.strip()]
                    else:
                        q["correct"] = corr.strip()
                pre.append(q)
            prefill_questions = pre
            st.success(f"Imported {len(prefill_questions)} questions from CSV.")
        except Exception as e:
            st.error("Failed to parse CSV: "+str(e))

    n = st.number_input("How many questions?", min_value=1, max_value=50, value=len(prefill_questions) if prefill_questions else 3, key="adm_n")
    questions = []
    for i in range(int(n)):
        p = prefill_questions[i] if prefill_questions and i<len(prefill_questions) else None
        q = q_block_admin(i, prefill=p)
        questions.append(q)

    st.subheader("Segmentation (optional) (up to 3)")
    segs = []
    for i in range(3):
        nm = st.text_input(f"Segment field {i+1} name (e.g., Class, Dept)", key=f"seg_name_{i}")
        vals = st.text_input(f"Allowed values (comma separated) for {nm}", key=f"seg_vals_{i}")
        if nm:
            segs.append({"name":nm,"allowed":[v.strip() for v in vals.split(",") if v.strip()]})

    if st.button("Save Poll (Admin)"):
        pid = make_code()
        cfg = {"questions": questions, "segments": segs, "session":{"present":False}}
        save_poll(pid, title.strip() or f"Untitled {pid}", desc.strip(), cfg)
        st.success(f"Saved poll with code: {pid}")
        st.code(f"?mode=vote&poll={pid}", language="text")
        st.code(f"?mode=results&poll={pid}", language="text")

def admin_control_ui():
    st.header("Admin Control Panel")
    polls = list_polls(100)
    sel = st.selectbox("Pick a poll to manage", [""]+[p["id"] for p in polls], key="admin_select_poll")
    if not sel:
        st.info("Select a poll to manage or create a new one under 'Create / Edit Poll'")
        return
    poll = load_poll(sel)
    if not poll:
        st.error("Poll not found")
        return
    st.subheader(f"Managing: {poll['title']} ({poll['id']})")
    cfg = poll["config"]
    session = cfg.get("session", {"present": False})
    cols = st.columns(4)
    with cols[0]:
        if st.button("Start Presenter Mode"):
            if cfg.get("questions"):
                session = {"present":True, "current_q":0, "started_at":now_ts(), "advance_at": now_ts()+int(cfg["questions"][0].get("time_seconds",20) or 20)}
                cfg["session"] = session; save_poll(sel,poll["title"],poll["description"],cfg)
                st.success("Presenter started")
    with cols[1]:
        if st.button("Previous Q"):
            if session.get("present"):
                cur = session.get("current_q",0)
                if cur>0:
                    cur-=1
                    session["current_q"]=cur
                    session["advance_at"]=now_ts()+int(cfg["questions"][cur].get("time_seconds",20) or 20)
                    cfg["session"]=session; save_poll(sel,poll["title"],poll["description"],cfg)
                    st.success("Moved to previous question")
    with cols[2]:
        if st.button("Next Q"):
            if session.get("present"):
                cur = session.get("current_q",0)
                if cur+1 < len(cfg.get("questions",[])):
                    cur+=1
                    session["current_q"]=cur
                    session["advance_at"]=now_ts()+int(cfg["questions"][cur].get("time_seconds",20) or 20)
                    cfg["session"]=session; save_poll(sel,poll["title"],poll["description"],cfg)
                    st.success("Advanced to next question")
                else:
                    session["present"]=False; cfg["session"]=session; save_poll(sel,poll["title"],poll["description"],cfg)
                    st.info("Reached end of questions")
    with cols[3]:
        if st.button("Stop Presenter"):
            session["present"]=False; cfg["session"]=session; save_poll(sel,poll["title"],poll["description"],cfg)
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
            # flatten answers into CSV
            rows = []
            for r in recs:
                base = {"ts":r["ts"], "user_tag":r["user_tag"], **{k:v for k,v in r["segment"].items()}}
                ans = r["answers"]
                for qid,qval in ans.items():
                    base[qid]=json.dumps(qval)
                rows.append(base)
            df = pd.DataFrame(rows)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name=f"responses_{sel}.csv", mime="text/csv")
        else:
            st.info("No responses to export")

def compute_scores(cfg, recs):
    # returns dict user_tag -> score, with breakdown
    scores = defaultdict(int)
    breakdown = defaultdict(list)
    for r in recs:
        uid = r["user_tag"]
        ans = r["answers"]
        s = 0
        for q in cfg.get("questions",[]):
            qid = q["id"]
            pts = int(q.get("points",0) or 0)
            if pts<=0: continue
            correct = q.get("correct")
            given = ans.get(qid)
            if correct is None or given is None: continue
            # scoring per type
            if q["type"] in ["multiple_choice","image_choice"]:
                if isinstance(given,str) and given.strip() and given.strip()==str(correct).strip():
                    s+=pts; breakdown[uid].append((qid,pts))
            elif q["type"]=="multiple_select":
                if isinstance(given,list):
                    # full points only if exactly matches (simpler)
                    if set([g.strip() for g in given])==set([c.strip() for c in correct]):
                        s+=pts; breakdown[uid].append((qid,pts))
            elif q["type"]=="ranking":
                # Borda-like: award points if positions match? simplified: full points if order matches
                if isinstance(given,list) and isinstance(correct,list):
                    if [g.strip() for g in given]==[c.strip() for c in correct]:
                        s+=pts; breakdown[uid].append((qid,pts))
            elif q["type"]=="scale":
                try:
                    if int(given)==int(correct):
                        s+=pts; breakdown[uid].append((qid,pts))
                except: pass
        scores[uid]=s
    return scores, breakdown

def maybe_advance(poll):
    # client-driven check for advance
    cfg = poll["config"]
    session = cfg.get("session",{})
    if not session.get("present"): return cfg
    if now_ts() >= int(session.get("advance_at", 0)):
        cur = session.get("current_q", 0)
        if cur+1 < len(cfg.get("questions",[])):
            cur+=1
            session["current_q"]=cur
            session["advance_at"]= now_ts() + int(cfg["questions"][cur].get("time_seconds",20) or 20)
            cfg["session"]=session; save_poll(poll["id"], poll["title"], poll["description"], cfg)
        else:
            session["present"]=False; cfg["session"]=session; save_poll(poll["id"], poll["title"], poll["description"], cfg)
    return cfg

# ----------------- Voting UI -----------------
def vote_view(poll):
    st.header(poll["title"])
    if poll["description"]: st.write(poll["description"])
    cfg = poll["config"]
    session = cfg.get("session", {"present":False})
    show_questions = cfg.get("questions",[])
    single_q = False
    time_left = None
    warn = False
    if session.get("present"):
        cfg = load_poll(poll["id"])["config"]  # reload
        session = cfg.get("session", session)
        cfg = maybe_advance(load_poll(poll["id"]))  # triggers advance if needed
        session = cfg.get("session")
        cur = session.get("current_q",0)
        show_questions = [cfg["questions"][cur]]
        single_q = True
        time_left = int(session.get("advance_at", now_ts()) - now_ts())
        if time_left <= 3 and time_left>0:
            warn = True
        st.write(f"Presenter mode — Question {cur+1} / {len(cfg.get('questions',[]))}")
        st.progress(int(((cur+1)/len(cfg.get('questions',[])))*100))
        st.metric("Time left", f"{time_left}s" if time_left is not None else "—")
        if st.checkbox("Auto-refresh (presenter mode)", key="auto_refresh_present", value=True):
            st.experimental_rerun()
    # Segments
    segvals = {}
    if cfg.get("segments"):
        st.subheader("Segmentation (optional)")
        for s in cfg["segments"]:
            if s.get("allowed"):
                v = st.selectbox(s["name"], ["(skip)"]+s["allowed"], key=f"seg_{s['name']}")
                segvals[s["name"]] = None if v=="(skip)" else v
            else:
                segvals[s["name"]] = st.text_input(s["name"], key=f"seg_{s['name']}")

    answers = {}
    st.subheader("Answer the question(s)")
    for q in show_questions:
        st.markdown(f"**{q['prompt']}**")
        qid = q["id"]
        qtype = q["type"]
        if qtype == "multiple_choice":
            # large button layout
            cols = st.columns(1)
            choice = st.radio("Select one:", q.get("options",[]), key=qid)
            answers[qid] = choice
        elif qtype == "multiple_select":
            sel = st.multiselect("Select any:", q.get("options",[]), key=qid)
            answers[qid]=sel
        elif qtype == "short_text":
            val = st.text_input("Short answer:", key=qid)
            answers[qid]=val
        elif qtype == "word_cloud":
            val = st.text_input("Type short phrase:", key=qid)
            answers[qid]=val.strip()
        elif qtype == "ranking":
            rem=list(q.get("options",[])); ranking=[]
            for k in range(len(q.get("options",[]))):
                pick = st.selectbox(f"Pick rank {k+1}", ["(choose)"]+rem, key=f"{qid}_rank_{k}")
                if pick!="(choose)" and pick in rem:
                    ranking.append(pick); rem.remove(pick)
            answers[qid]=ranking
        elif qtype == "scale":
            val = st.slider("Choose", min_value=int(q.get("min",1)), max_value=int(q.get("max",5)), step=int(q.get("step",1)), key=qid)
            answers[qid]=val
        elif qtype == "image_choice":
            labels=[o.get("label") for o in q.get("options",[])]
            choice = st.radio("Select one:", labels, key=qid)
            # show images
            cols = st.columns(min(4,len(labels) if labels else 1))
            for idx,o in enumerate(q.get("options",[])):
                with cols[idx%len(cols)]:
                    if o.get("image"): st.image(o.get("image"), caption=o.get("label"), use_column_width=True)
                    else: st.write(o.get("label"))
            answers[qid]=choice
        st.divider()
    # show immediate correct answer if configured and user submitted
    if st.button("Submit answers", key="submit_answers_btn"):
        # validate requireds
        missing=[]
        for q in show_questions:
            if not q.get("required",True): continue
            v = answers.get(q["id"])
            if q["type"]=="ranking":
                if not v or len(v)!=len(q.get("options",[])):
                    missing.append(q["prompt"]+" (complete ranking)")
            elif q["type"]=="multiple_select":
                if not v: missing.append(q["prompt"])
            elif q["type"]=="word_cloud" or q["type"]=="short_text":
                if not v: missing.append(q["prompt"])
            else:
                if v is None or v==[]: missing.append(q["prompt"])
        if missing:
            st.error("Complete required: \n- " + "\n- ".join(missing))
        else:
            save_response(poll["id"], anon_tag(), segvals, answers)
            st.success("Response recorded!")
            # if show immediate answer on question level, display
            for q in show_questions:
                if q.get("show_answer_immediate") and q.get("correct") is not None:
                    st.info(f"Correct answer for: {q['prompt']} -> {q.get('correct')}")
            st.balloons()
            st.experimental_rerun()

# ----------------- Results UI -----------------
def results_view(poll):
    st.header("Results — " + poll["title"])
    cfg = poll["config"]
    recs = load_responses(poll["id"])
    st.caption(f"Total responses: {len(recs)}")
    if not recs:
        st.info("No responses yet")
        return
    # Segment filters
    segs=[s["name"] for s in cfg.get("segments",[])]
    active={}
    if segs:
        cols=st.columns(len(segs))
        for i,s in enumerate(segs):
            with cols[i]:
                opts = sorted(set([r["segment"].get(s) for r in recs if r["segment"].get(s)]))
                sel = st.selectbox(s, ["(All)"]+opts, key=f"res_seg_{s}")
                if sel!="(All)": active[s]=sel
    def passes(r):
        for k,v in active.items():
            if r["segment"].get(k)!=v: return False
        return True
    filtered=[r for r in recs if passes(r)]
    st.caption(f"Showing {len(filtered)} responses after filters")
    # Per-question aggregation
    for q in cfg.get("questions",[]):
        st.subheader(q["prompt"])
        qid=q["id"]; qtype=q["type"]
        answers=[r["answers"].get(qid) for r in filtered if qid in r["answers"]]
        if qtype in ["multiple_choice","image_choice"]:
            counts=Counter([a for a in answers if a is not None])
            df=pd.DataFrame({"option":list(counts.keys()),"count":list(counts.values())}).sort_values("count",ascending=False)
            st.bar_chart(df.set_index("option")); st.dataframe(df)
        elif qtype=="multiple_select":
            flat=[]
            for a in answers:
                if isinstance(a,list): flat.extend(a)
            counts=Counter(flat)
            df=pd.DataFrame({"option":list(counts.keys()),"count":list(counts.values())}).sort_values("count",ascending=False)
            st.bar_chart(df.set_index("option")); st.dataframe(df)
        elif qtype=="word_cloud" or qtype=="short_text":
            norm=[str(a).strip().lower() for a in answers if isinstance(a,str) and a.strip()]
            counts=Counter(norm)
            df=pd.DataFrame({"text":list(counts.keys()),"count":list(counts.values())}).sort_values("count",ascending=False)
            st.dataframe(df)
        elif qtype=="ranking":
            scores=Counter(); N=len(q.get("options",[]))
            for r in answers:
                if isinstance(r,list) and len(r)==N:
                    for rank,opt in enumerate(r):
                        scores[opt]+= (N-rank)
            df=pd.DataFrame({"option":list(scores.keys()),"score":list(scores.values())}).sort_values("score",ascending=False)
            st.bar_chart(df.set_index("option")); st.dataframe(df)
        elif qtype=="scale":
            nums=[a for a in answers if isinstance(a,(int,float))]
            if nums:
                st.metric("Average",f"{np.mean(nums):.2f}"); st.metric("Median",f"{np.median(nums):.2f}"); st.line_chart(pd.DataFrame({"value":nums}))
        st.divider()
    # Leaderboard if scoring configured
    scores,breakdown = compute_scores(cfg, filtered)
    if scores:
        st.subheader("Leaderboard")
        rows=[{"user_tag":k,"score":v} for k,v in scores.items()]
        df=pd.DataFrame(rows).sort_values("score",ascending=False)
        st.dataframe(df.head(10))
        st.download_button("Download leaderboard CSV", data=df.to_csv(index=False).encode("utf-8"), file_name=f"leaderboard_{poll['id']}.csv")

# ----------------- Main app -----------------
def main():
    st.set_page_config(page_title="Poll Maker — Enhanced", layout="wide")
    init_db()
    q = st.experimental_get_query_params()
    mode = q.get("mode", [""])[0]
    pid = q.get("poll", [""])[0].upper()

    with st.sidebar:
        st.title("Poll Maker")
        nav = st.radio("Navigate", ["Home","Admin","Create","Join","Vote","Results"], index=0 if not mode else (["","admin","create","join","vote","results"].index(mode)))
        if nav=="Home": st.experimental_set_query_params()
        elif nav=="Admin": st.experimental_set_query_params(mode="admin")
        elif nav=="Create": st.experimental_set_query_params(mode="create")
        elif nav=="Join": st.experimental_set_query_params(mode="join")
        elif nav=="Vote": st.experimental_set_query_params(mode="vote", poll=pid)
        elif nav=="Results": st.experimental_set_query_params(mode="results", poll=pid)
        st.markdown("---")
        st.caption("Admin password required for Admin/Create. Presenter mode supports timed auto-advance.")

    # admin auth
    is_admin=False
    if mode in ["admin","create"]:
        pwd = st.text_input("Admin password", type="password", key="pwd_input")
        if pwd:
            if pwd==ADMIN_PASSWORD:
                is_admin=True; st.success("Authenticated as admin")
            else:
                st.error("Invalid password")

    if mode=="create":
        if is_admin: create_poll_admin_ui()
        else: st.error("Authenticate in Admin with password to create polls.")
    elif mode=="admin":
        if is_admin: admin_control_ui()
        else: st.info("Enter admin password to access controls.")
    elif mode=="join":
        code = st.text_input("Enter poll code", key="join_code")
        if st.button("Open poll"):
            st.experimental_set_query_params(mode="vote", poll=code.strip().upper()); st.experimental_rerun()
    elif mode=="vote" and pid:
        poll = load_poll(pid)
        if poll: vote_view(poll)
        else: st.error("Poll not found")
    elif mode=="results" and pid:
        poll = load_poll(pid)
        if poll: results_view(poll)
        else: st.error("Poll not found")
    else:
        st.title("Interactive Poll Maker — Home")
        st.write("Create polls, run presenter-mode quizzes, and collect responses. Use Admin to manage polls.")

if __name__ == '__main__':
    main()
