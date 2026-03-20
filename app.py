import streamlit as st
import random
import re

from document_processor import extract_text
from embeddings import create_chunks
from rag_graph import store_vectors, retrieve_docs
from evaluation import evaluate_answer
from question_generator import generate_question
from question_utils import extract_questions


st.set_page_config(
    page_title="AI Mock Interview",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────
#               CUSTOM CSS
# ─────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Lora:wght@600;700&display=swap');

/* ── Design tokens ── */
:root {
    --bg:         #f4f6fb;
    --white:      #ffffff;
    --border:     #e2e6f0;
    --accent:     #3b6cf7;
    --accent-lt:  #eef2fe;
    --success:    #16a34a;
    --success-lt: #dcfce7;
    --warning:    #b45309;
    --warning-lt: #fef3c7;
    --danger:     #dc2626;
    --danger-lt:  #fee2e2;
    --text:       #111827;
    --text-2:     #4b5563;
    --muted:      #9ca3af;
    --shadow-sm:  0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow:     0 4px 16px rgba(0,0,0,0.07), 0 1px 4px rgba(0,0,0,0.05);
    --r:          12px;
    --r-sm:       8px;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
    font-size: 16px;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 5rem; max-width: 880px; margin: auto; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

/* ────── SIDEBAR ────── */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.6rem 1.1rem; }

.sb-logo {
    font-family: 'Lora', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
    padding-bottom: 1.2rem;
    margin-bottom: 1.4rem;
    border-bottom: 1px solid var(--border);
}

.sb-label {
    font-size: 0.92rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.3rem 0 0.6rem;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.85rem 1rem;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    margin-bottom: 0.45rem;
}
.stat-row .sl { font-size: 1.05rem; color: var(--text-2); }
.stat-row .sv { font-family: 'Lora', serif; font-size: 1.25rem; font-weight: 700; color: var(--text); }
.sv.blue  { color: var(--accent);  }
.sv.green { color: var(--success); }

.diff-pill {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 99px;
    font-size: 0.96rem;
    font-weight: 600;
    margin-top: 0.45rem;
}
.dp-beginner     { background: var(--success-lt); color: var(--success); }
.dp-intermediate { background: var(--warning-lt); color: var(--warning); }
.dp-advanced     { background: var(--danger-lt);  color: var(--danger);  }

.hist-row {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 1rem;
}
.hist-row .hq { color: var(--text-2); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-row .hs { font-weight: 700; flex-shrink: 0; }

/* ────── PAGE HEADER ────── */
.page-header {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.6rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
    box-shadow: var(--shadow-sm);
}
.ph-icon  { font-size: 2.4rem; flex-shrink: 0; }
.ph-title {
    font-family: 'Lora', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.2rem;
}
.ph-sub   { font-size: 1rem; color: var(--muted); margin: 0; }

/* ────── UPLOAD ZONE ────── */
.upload-box {
    background: var(--white);
    border: 2px dashed var(--border);
    border-radius: var(--r);
    padding: 2.6rem 2rem;
    text-align: center;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.upload-box:hover { border-color: var(--accent); }
.upload-box .ub-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.upload-box h3 {
    font-family: 'Lora', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.3rem;
}
.upload-box p { font-size: 0.95rem; color: var(--muted); margin: 0; }

/* ────── PROGRESS ────── */
.prog-wrap {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 1rem 1.3rem;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow-sm);
}
.prog-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--muted);
    margin-bottom: 0.55rem;
}
.prog-track {
    height: 6px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 99px;
    transition: width 0.4s ease;
}

/* ────── QUESTION CARD ────── */
.q-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: var(--r);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.3rem;
    box-shadow: var(--shadow-sm);
}
.q-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.7rem;
}
.tag {
    background: var(--accent-lt);
    color: var(--accent);
    border-radius: 99px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 0.2rem 0.6rem;
    text-transform: uppercase;
}
.q-text {
    font-family: 'Lora', serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.65;
    margin: 0;
}

/* ────── ANSWER LABEL ────── */
.ans-label {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.1rem 0 0.4rem;
}

/* ────── FEEDBACK CARD ────── */
.fb-card {
    background: #f0f4ff;
    border: 1px solid #c7d5fc;
    border-radius: var(--r);
    padding: 1.4rem 1.6rem;
    margin-top: 1.1rem;
}
.fb-title {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 0.65rem;
}
.fb-body {
    font-size: 1rem;
    line-height: 1.75;
    color: var(--text-2);
    margin: 0;
}

/* ────── SCORE BADGE ────── */
.sc-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.38rem 1rem;
    border-radius: 99px;
    font-size: 0.92rem;
    font-weight: 700;
    margin-top: 0.9rem;
}
.sc-high { background: var(--success-lt); color: var(--success); }
.sc-mid  { background: var(--warning-lt); color: var(--warning); }
.sc-low  { background: var(--danger-lt);  color: var(--danger);  }

/* ────── REPORT CARD ────── */
.report-wrap {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 2rem 2.2rem;
    margin-top: 1rem;
    box-shadow: var(--shadow);
    text-align: center;
}
.rw-icon  { font-size: 3rem; margin-bottom: 0.4rem; }
.rw-score {
    font-family: 'Lora', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    margin: 0.2rem 0;
}
.rw-sub { font-size: 0.97rem; color: var(--muted); margin-bottom: 1.5rem; }

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
    margin-bottom: 1.3rem;
}
.metric-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 1rem 0.8rem;
}
.mb-val {
    font-family: 'Lora', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
}
.mb-lbl { font-size: 0.82rem; color: var(--muted); margin-top: 0.15rem; }

.verdict-box {
    border-radius: var(--r-sm);
    padding: 1rem 1.2rem;
    font-size: 1rem;
    font-weight: 500;
    line-height: 1.6;
}
.vd-ex { background: var(--success-lt); color: var(--success); }
.vd-go { background: var(--warning-lt); color: var(--warning); }
.vd-nd { background: var(--danger-lt);  color: var(--danger);  }

/* ────── BUTTONS ────── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 2px 8px rgba(59,108,247,0.22) !important;
    transition: opacity 0.15s, transform 0.1s !important;
}
.stButton > button:hover  { opacity: 0.9 !important; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled { opacity: 0.38 !important; cursor: not-allowed !important; box-shadow: none !important; }

.btn-danger .stButton > button {
    background: transparent !important;
    color: var(--danger) !important;
    border: 1px solid #fca5a5 !important;
    box-shadow: none !important;
}
.btn-danger .stButton > button:hover { background: var(--danger-lt) !important; }

/* ────── INPUTS ────── */
textarea, [data-testid="stTextArea"] textarea {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--r-sm) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1rem !important;
    min-height: 120px !important;
}
textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,108,247,0.09) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--r-sm) !important;
    font-size: 1rem !important;
}
[data-testid="stAlert"] {
    border-radius: var(--r-sm) !important;
    font-size: 0.97rem !important;
}
hr { border-color: var(--border) !important; margin: 1.4rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#            SESSION STATE
# ─────────────────────────────────────────

for k, v in {
    "questions": [],
    "current_q": 0,
    "answered": False,
    "scores": [],
    "feedback_history": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────
#            SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sb-logo">🎯 InterviewAI</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Settings</div>', unsafe_allow_html=True)
    difficulty = st.selectbox(
        "Difficulty Level",
        ["Beginner", "Intermediate", "Advanced"],
        help="Sets the complexity of generated questions"
    )
    dp_map = {
        "Beginner": "dp-beginner",
        "Intermediate": "dp-intermediate",
        "Advanced": "dp-advanced"
    }
    st.markdown(
        f'<span class="diff-pill {dp_map[difficulty]}">{difficulty}</span>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sb-label">Session Stats</div>', unsafe_allow_html=True)

    total_q    = len(st.session_state.questions)
    answered_n = st.session_state.current_q
    avg_sc     = (
        round(sum(st.session_state.scores) / len(st.session_state.scores), 1)
        if st.session_state.scores else "—"
    )

    st.markdown(f"""
    <div class="stat-row">
        <span class="sl">Total Questions</span>
        <span class="sv blue">{total_q}</span>
    </div>
    <div class="stat-row">
        <span class="sl">Answered</span>
        <span class="sv">{answered_n}</span>
    </div>
    <div class="stat-row">
        <span class="sl">Avg Score</span>
        <span class="sv green">{avg_sc}</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.scores:
        st.markdown('<div class="sb-label">Score History</div>', unsafe_allow_html=True)
        for i, (q, sc) in enumerate(zip(
            st.session_state.questions[:len(st.session_state.scores)],
            st.session_state.scores
        )):
            color = (
                "var(--success)" if sc >= 8
                else "var(--warning)" if sc >= 5
                else "var(--danger)"
            )
            short_q = q[:40] + "…" if len(q) > 40 else q
            st.markdown(f"""
            <div class="hist-row">
                <span class="hq">Q{i+1}. {short_q}</span>
                <span class="hs" style="color:{color}">{sc}/10</span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#            PAGE HEADER
# ─────────────────────────────────────────

st.markdown("""
<div class="page-header">
    <div class="ph-icon">🎯</div>
    <div>
        <div class="ph-title">AI Mock Interview</div>
        <p class="ph-sub">Upload study material · Generate questions · Get AI-powered feedback</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#            FILE UPLOAD
# ─────────────────────────────────────────

if len(st.session_state.questions) == 0:
    st.markdown("""
    <div class="upload-box">
        <div class="ub-icon">📄</div>
        <h3>Upload your study materials</h3>
        <p>PDF, DOCX, and TXT supported — multiple files accepted</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        with st.spinner("Processing your documents…"):
            text_data = ""
            for file in uploaded_files:
                text_data += extract_text(file)

            existing_questions = extract_questions(text_data)

            if len(existing_questions) >= 3:
                questions = existing_questions
                st.info("Questions detected in the document — using them directly.")
            else:
                chunks = create_chunks(text_data)
                store_vectors(chunks)
                docs = retrieve_docs("interview questions")
                questions = []
                for doc in docs:
                    q = generate_question(doc, difficulty)
                    questions.append(q)
                st.info(f"Generating {difficulty} level questions from your material.")

            questions = list(set(questions))
            random.shuffle(questions)

            st.session_state.questions        = questions
            st.session_state.current_q        = 0
            st.session_state.answered         = False
            st.session_state.scores           = []
            st.session_state.feedback_history = []

        st.success("Documents processed — your interview is ready!")
        st.rerun()


# ─────────────────────────────────────────
#            PROGRESS BAR
# ─────────────────────────────────────────

if st.session_state.questions:
    total   = len(st.session_state.questions)
    current = st.session_state.current_q
    pct     = min(int((current / total) * 100), 100)

    st.markdown(f"""
    <div class="prog-wrap">
        <div class="prog-meta">
            <span>Progress</span>
            <span>{min(current, total)} of {total} questions</span>
        </div>
        <div class="prog-track">
            <div class="prog-fill" style="width:{pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#            QUESTION DISPLAY
# ─────────────────────────────────────────

qs = st.session_state.questions
cq = st.session_state.current_q

if qs and cq < len(qs):
    question = qs[cq]

    st.markdown(f"""
    <div class="q-card">
        <div class="q-meta">
            Question {cq + 1} of {len(qs)}
            <span class="tag">{difficulty}</span>
        </div>
        <p class="q-text">{question}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ans-label">Your Answer</div>', unsafe_allow_html=True)
    answer = st.text_area(
        "Answer",
        placeholder="Write your answer here — be as detailed as possible…",
        key=f"answer_box_{cq}",
        label_visibility="collapsed"
    )

    col_sub, col_next, _ = st.columns([1, 1, 4])
    with col_sub:
        submit = st.button("Submit", use_container_width=True)
    with col_next:
        if st.button("Next →", disabled=not st.session_state.answered, use_container_width=True):
            st.session_state.current_q += 1
            st.session_state.answered   = False
            st.rerun()

    if submit:
        if not answer.strip():
            st.warning("Please write an answer before submitting.")
        else:
            with st.spinner("Evaluating your answer…"):
                feedback = evaluate_answer(answer)

            score_match = re.search(r"Score:\s*(\d+)", feedback)
            score       = int(score_match.group(1)) if score_match else None

            if score is not None:
                st.session_state.scores.append(score)
                sc_cls  = "sc-high" if score >= 8 else "sc-mid" if score >= 5 else "sc-low"
                sc_icon = "✓" if score >= 8 else "~" if score >= 5 else "✗"
                score_html = f'<div class="sc-badge {sc_cls}">{sc_icon} Score: {score} / 10</div>'
            else:
                score_html = ""

            st.session_state.feedback_history.append(feedback)

            st.markdown(f"""
            <div class="fb-card">
                <div class="fb-title">AI Feedback</div>
                <p class="fb-body">{feedback}</p>
                {score_html}
            </div>
            """, unsafe_allow_html=True)

            st.session_state.answered = True


# ─────────────────────────────────────────
#            FINAL REPORT
# ─────────────────────────────────────────

if qs and cq >= len(qs):
    scores = st.session_state.scores

    if scores:
        avg  = sum(scores) / len(scores)
        high = max(scores)
        low  = min(scores)

        if avg >= 8:
            icon, vd_cls = "🏆", "vd-ex"
            msg = "Excellent performance! You've demonstrated strong command of the subject."
        elif avg >= 5:
            icon, vd_cls = "📈", "vd-go"
            msg = "Good attempt. A few areas need attention — keep practicing."
        else:
            icon, vd_cls = "📚", "vd-nd"
            msg = "Needs improvement. Review the key concepts and try again."

        st.markdown(f"""
        <div class="report-wrap">
            <div class="rw-icon">{icon}</div>
            <div class="rw-score">{round(avg, 1)} / 10</div>
            <div class="rw-sub">Average score across {len(scores)} question{"s" if len(scores) != 1 else ""}</div>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="mb-val">{len(scores)}</div>
                    <div class="mb-lbl">Answered</div>
                </div>
                <div class="metric-box">
                    <div class="mb-val" style="color:var(--success)">{high}</div>
                    <div class="mb-lbl">Best Score</div>
                </div>
                <div class="metric-box">
                    <div class="mb-val" style="color:var(--danger)">{low}</div>
                    <div class="mb-lbl">Lowest Score</div>
                </div>
            </div>
            <div class="verdict-box {vd_cls}">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────
#            RESTART
# ─────────────────────────────────────────

if qs:
    st.markdown("---")
    st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
    if st.button("↺ Restart Interview"):
        for k, v in {
            "questions": [], "current_q": 0,
            "answered": False, "scores": [], "feedback_history": []
        }.items():
            st.session_state[k] = v
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)