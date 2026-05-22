import os
import random
import time

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Credit Risk Analyzer', layout='wide', page_icon='💳')

# --- Model + artifacts (best-effort load) ---
rf_model = None
try:
    if os.path.exists('random_forest_model.pkl'):
        rf_model = joblib.load('random_forest_model.pkl')
except Exception:
    rf_model = None

model = None
scaler = None
feature_columns = None
try:
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
except Exception:
    # keep going; UI still works for what-if and heuristics
    model = model or None

# dataset for dropdown options
raw_df = pd.DataFrame()
if os.path.exists('credit_risk_dataset.csv'):
    try:
        raw_df = pd.read_csv('credit_risk_dataset.csv')
    except Exception:
        raw_df = pd.DataFrame()

# --- Helpers ---

def get_risk_status(score: float):
    if score < 0.35:
        return 'Low Risk', '#2ef0b3'
    if score < 0.65:
        return 'Medium Risk', '#ffd55d'
    return 'High Risk', '#ff6a6a'


def build_gauge(score: float, label: str, color: str):
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=score * 100,
        number={'suffix': '%', 'font': {'color': '#0b1b3a', 'size': 28}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'bgcolor': 'rgba(0,0,0,0)',
            'steps': [
                {'range': [0, 40], 'color': '#e6eef9'},
                {'range': [40, 70], 'color': '#d6e4ff'},
                {'range': [70, 100], 'color': '#cde6ff'},
            ],
            'threshold': {'line': {'color': color, 'width': 4}, 'value': score * 100},
        },
        title={'text': label, 'font': {'color': '#0b1b3a', 'size': 14}},
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, b=10, l=10, r=10), height=260,
                      transition={'duration': 450, 'easing': 'cubic-in-out'})
    return fig


def build_bar_chart(metrics):
    fig = go.Figure(data=[go.Bar(x=list(metrics.keys()), y=list(metrics.values()), marker_color=['#4b8cff', '#8bc4ff', '#7af3d0', '#ffcd76'])])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, b=30), height=300,
                      transition={'duration': 450, 'easing': 'cubic-in-out'})
    fig.update_yaxes(range=[0, 100])
    return fig


def build_pie_chart(breakdown):
    fig = go.Figure(data=[go.Pie(labels=list(breakdown.keys()), values=list(breakdown.values()), hole=0.55)])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=320,
                      transition={'duration': 450, 'easing': 'cubic-in-out'})
    return fig


def build_sparkline(values, color='#5cf2d1'):
    fig = go.Figure(go.Scatter(x=list(range(len(values))), y=values, mode='lines', line=dict(color=color, width=3), fill='tozeroy'))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=6, b=6), height=120,
                      transition={'duration': 350, 'easing': 'cubic-in-out'})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def build_progress(label: str, value: float, color: str):
    width = int(value * 100)
    return f'''<div class="mini-card"><div class="mini-title">{label}</div><div class="progress-shell"><div class="progress-fill" style="width:{width}%; background: {color};"></div></div><div class="progress-value">{width}%</div></div>'''

# --- CSS / Theme ---
base_css = """
<style>
:root{ --bg:#050816; --panel:rgba(13,24,58,0.72); --text:#e5efff; }
html, body { background: linear-gradient(180deg,#040811 0%, #050816 100%); color:var(--text); font-family:Inter, sans-serif }
.stApp { background: transparent }
.glass-panel { background:var(--panel); border-radius:18px; padding:18px; border:1px solid rgba(255,255,255,0.04); }
.animal-card{display:inline-flex;align-items:center;justify-content:center;width:110px;height:110px;border-radius:18px;font-size:56px;transition:transform .25s ease, box-shadow .25s ease}
.animal-turtle{background:linear-gradient(135deg,#e6fff5,#d0fff0);box-shadow:0 8px 30px rgba(46,240,179,0.12)}
.animal-dog{background:linear-gradient(135deg,#fff7e6,#fff1d0);box-shadow:0 8px 30px rgba(255,213,93,0.12);transform:scale(1.03)}
.animal-tiger{background:linear-gradient(135deg,#ffecec,#ffd8d8);box-shadow:0 10px 36px rgba(255,100,100,0.16);transform:scale(1.06);animation:tigerPulse 1.6s ease-in-out infinite}
@keyframes tigerPulse{0%{transform:scale(1.04)}50%{transform:scale(1.08)}100%{transform:scale(1.04)}}
.mini-card{border-radius:12px;background:rgba(255,255,255,0.03);padding:12px}
.mini-title{color:#9ab1ff;font-size:12px}
.metric-value{font-size:28px;font-weight:700}

/* Animated header title */
.animated-title{font-size:48px;font-weight:800;letter-spacing:-1px;line-height:1;display:inline-block;background:linear-gradient(90deg,#6d53ff,#39f2be,#7f8dff);-webkit-background-clip:text;background-clip:text;color:transparent;padding:6px 10px;border-radius:8px;transform:translateY(-8px);opacity:0;animation:titleEnter 900ms cubic-bezier(.2,.9,.2,1) forwards}
.animated-sub{color:#9bb2ff;margin-top:6px;opacity:0;transform:translateY(6px);animation:subEnter 900ms 160ms cubic-bezier(.2,.9,.2,1) forwards}
@keyframes titleEnter{0%{opacity:0;transform:translateY(-18px) scale(.98)}60%{opacity:1;transform:translateY(2px) scale(1.01)}100%{opacity:1;transform:translateY(0) scale(1)}}
@keyframes subEnter{0%{opacity:0;transform:translateY(12px)}100%{opacity:1;transform:translateY(0)}}

/* Floating animated text pill */
.floating-banner{margin-top:18px;display:flex;flex-wrap:wrap;gap:14px;justify-content:center;align-items:center;padding:18px 20px;border-radius:32px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);backdrop-filter:blur(18px);overflow:hidden}
.floating-banner span{display:inline-block;padding:10px 18px;border-radius:999px;background:rgba(255,255,255,0.08);color:#edf7ff;font-size:14px;font-weight:600;letter-spacing:.2px;animation:floatText 5s ease-in-out infinite;box-shadow:0 12px 22px rgba(0,0,0,0.08)}
.floating-banner span:nth-child(2){animation-delay:0.75s}
.floating-banner span:nth-child(3){animation-delay:1.5s}
@keyframes floatText{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}

/* shimmering gradient sweep */
.animated-title{position:relative}
.animated-title::after{content:'';position:absolute;left:0;top:0;right:0;bottom:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.18),transparent);mix-blend-mode:overlay;animation:shimmer 2.2s linear infinite}
@keyframes shimmer{0%{transform:translateX(-110%)}100%{transform:translateX(110%)}}
</style>
"""
st.markdown(base_css, unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="glass-panel" style="margin-bottom:16px; display:flex;justify-content:space-between;align-items:center;position:relative">', unsafe_allow_html=True)
st.markdown('<div><div class="animated-title">Credit Risk Analyzer</div><div class="animated-sub">Enter your details and get real-time AI risk prediction</div><div class="floating-banner"><span>Enter your details</span><span>Manual borrower input</span><span>Instant risk preview</span></div></div>', unsafe_allow_html=True)
# theme switch placeholder (kept in sidebar below)
st.markdown('</div>', unsafe_allow_html=True)

# --- Sidebar inputs (manual, realtime) ---
with st.sidebar:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('### Enter your details')
    model_choice = st.selectbox('Model', ['Logistic Regression', 'Random Forest'])
    if model_choice == 'Random Forest' and rf_model is None:
        st.warning('Random Forest model missing — using Logistic Regression')
        model_choice = 'Logistic Regression'

    theme = st.selectbox('Theme', ['Dark', 'Light'])
    live_update = st.checkbox('Live update (auto analyze)', value=True)
    preset = st.selectbox('Preset', ['Default', 'Conservative', 'Aggressive'])

    if preset == 'Conservative':
        defaults = {'age': 45, 'income': 90000, 'loan': 8000, 'int_rate': 6.0, 'dti': 12.0, 'emp_len': 12, 'cred': 720}
    elif preset == 'Aggressive':
        defaults = {'age': 26, 'income': 35000, 'loan': 25000, 'int_rate': 18.0, 'dti': 42.0, 'emp_len': 2, 'cred': 420}
    else:
        defaults = {'age': 33, 'income': 50000, 'loan': 12000, 'int_rate': 10.0, 'dti': 20.0, 'emp_len': 5, 'cred': 660}

    if not live_update:
        form = st.form('input_form')
        with form:
            age = st.slider('Age', 18, 100, defaults['age'], help='Borrower age')
            income = st.number_input('Annual Income (₹)', min_value=0, max_value=100000000, value=defaults['income'], step=1000, format='%d', help='Gross annual income')
            loan_amount = st.slider('Loan Amount (₹)', 100, 2000000, defaults['loan'], step=1000, help='Requested loan amount')
            credit_score_input = st.slider('Credit Score', 300, 900, defaults['cred'], help='Bureau credit score')
            interest_rate = st.slider('Interest Rate (%)', 0.0, 50.0, defaults['int_rate'], step=0.1)
            dti = st.slider('Debt-to-Income (%)', 0.0, 100.0, defaults['dti'], step=0.5)
            employment_type = st.selectbox('Employment Type', ['Salaried', 'Self-employed', 'Unemployed', 'Retired'])
            has_existing = st.checkbox('Has existing loans?')
            existing_count = st.number_input('Existing loans', min_value=0, max_value=20, value=0) if has_existing else 0
            emp_length = st.number_input('Employment length (yrs)', min_value=0, max_value=60, value=defaults['emp_len'])
            cred_hist = st.number_input('Credit history (yrs)', min_value=0, max_value=60, value=8)
            home_ownership = st.selectbox('Home Ownership', raw_df['person_home_ownership'].dropna().unique() if not raw_df.empty else ['RENT', 'OWN'])
            loan_intent = st.selectbox('Loan Intent', raw_df['loan_intent'].dropna().unique() if not raw_df.empty else ['PERSONAL', 'EDUCATION', 'MEDICAL'])
            loan_grade = st.selectbox('Loan Grade', raw_df['loan_grade'].dropna().unique() if not raw_df.empty else ['A', 'B', 'C'])
            default_on_file = st.selectbox('Default on file', raw_df['cb_person_default_on_file'].dropna().unique() if not raw_df.empty else ['No', 'Yes'])

            what_if = st.slider('What-if: increase loan by (%)', 0, 200, 0, step=5, help='Simulate loan increase')

            submit = form.form_submit_button('Analyze Risk')
    else:
        age = st.slider('Age', 18, 100, defaults['age'], help='Borrower age')
        income = st.number_input('Annual Income (₹)', min_value=0, max_value=100000000, value=defaults['income'], step=1000, format='%d', help='Gross annual income')
        loan_amount = st.slider('Loan Amount (₹)', 100, 2000000, defaults['loan'], step=1000, help='Requested loan amount')
        credit_score_input = st.slider('Credit Score', 300, 900, defaults['cred'], help='Bureau credit score')
        interest_rate = st.slider('Interest Rate (%)', 0.0, 50.0, defaults['int_rate'], step=0.1)
        dti = st.slider('Debt-to-Income (%)', 0.0, 100.0, defaults['dti'], step=0.5)
        employment_type = st.selectbox('Employment Type', ['Salaried', 'Self-employed', 'Unemployed', 'Retired'])
        has_existing = st.checkbox('Has existing loans?')
        existing_count = st.number_input('Existing loans', min_value=0, max_value=20, value=0) if has_existing else 0
        emp_length = st.number_input('Employment length (yrs)', min_value=0, max_value=60, value=defaults['emp_len'])
        cred_hist = st.number_input('Credit history (yrs)', min_value=0, max_value=60, value=8)
        home_ownership = st.selectbox('Home Ownership', raw_df['person_home_ownership'].dropna().unique() if not raw_df.empty else ['RENT', 'OWN'])
        loan_intent = st.selectbox('Loan Intent', raw_df['loan_intent'].dropna().unique() if not raw_df.empty else ['PERSONAL', 'EDUCATION', 'MEDICAL'])
        loan_grade = st.selectbox('Loan Grade', raw_df['loan_grade'].dropna().unique() if not raw_df.empty else ['A', 'B', 'C'])
        default_on_file = st.selectbox('Default on file', raw_df['cb_person_default_on_file'].dropna().unique() if not raw_df.empty else ['No', 'Yes'])

        what_if = st.slider('What-if: increase loan by (%)', 0, 200, 0, step=5, help='Simulate loan increase')

        submit = True

    st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction & UI updates ---
if submit:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    with st.spinner('Updating prediction...'):
        time.sleep(0.45)

        loan_adj = int(loan_amount * (1 + (what_if / 100.0)))
        loan_percent_income_ratio = dti / 100.0

        user = {
            'person_age': age,
            'person_income': income,
            'person_emp_length': emp_length,
            'loan_amnt': loan_adj,
            'loan_int_rate': interest_rate,
            'loan_percent_income': loan_percent_income_ratio,
            'cb_person_cred_hist_length': cred_hist,
            'person_home_ownership': home_ownership,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'cb_person_default_on_file': default_on_file,
        }

        df_in = pd.DataFrame([user])
        cat_feats = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        df_enc = pd.get_dummies(df_in, columns=cat_feats, drop_first=True)
        if feature_columns is not None:
            for c in feature_columns:
                if c not in df_enc.columns:
                    df_enc[c] = 0
            df_enc = df_enc[feature_columns]
            try:
                df_enc[['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']] = scaler.transform(df_enc[['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']])
            except Exception:
                pass

        chosen = model if model_choice == 'Logistic Regression' or rf_model is None else rf_model
        if chosen is not None:
            try:
                pred = chosen.predict(df_enc)[0]
                prob = chosen.predict_proba(df_enc)[0]
                risk_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
            except Exception:
                pred = 0
                risk_prob = 0.42
        else:
            pred = 0
            risk_prob = 0.42

        risk_level, risk_color = get_risk_status(risk_prob)
        # display the user input credit score as the main credit score
        credit_score = int(credit_score_input)
        approval = min(100, max(15, int((1 - risk_prob) * 100 + (credit_score - 600) / 5 + 10)))

        breakdown = {
            'Loan Size': min(100, loan_adj / 30000 * 100),
            'Income': max(0, 100 - min(100, income / 4000)),
            'Credit History': max(0, 100 - min(100, cred_hist * 5)),
            'Interest Rate': min(100, interest_rate * 2.2),
            'Default': 30 if str(default_on_file).lower() == 'yes' else 8,
        }
        breakdown_norm = {k: float(v) for k, v in breakdown.items()}
        metrics = {
            'Credit Score': credit_score,
            'Approval Likelihood': approval,
            'Debt Load': min(100, dti * 1.2),
            'Stability': max(0, min(100, emp_length * 3 + cred_hist * 2 - existing_count * 4)),
        }

    # Results header with animal
    animal_html = "<div class='animal-card animal-turtle'>🐢</div>" if risk_level == 'Low Risk' else ("<div class='animal-card animal-dog'>🐕</div>" if risk_level == 'Medium Risk' else "<div class='animal-card animal-tiger'>🐅</div>")
    st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;gap:20px;'><div><h2 style='margin:0'>Risk summary</h2><p style='margin:4px 0 0;color:#9bb2ff'>Live AI risk preview</p></div><div style='display:flex;gap:12px;align-items:center'>{animal_html}<div style='display:flex;flex-direction:column;'><span style='padding:6px 10px;border-radius:12px;background:rgba(255,255,255,0.04)'>{risk_level}</span><span style='padding:6px 10px;border-radius:12px;background:rgba(255,255,255,0.03)'>Approval {approval}%</span></div></div></div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # metric cards
    risk_state_class = 'risk-state-low' if risk_level == 'Low Risk' else 'risk-state-medium' if risk_level == 'Medium Risk' else 'risk-state-high'
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        st.markdown(f"<div class='glass-panel' style='padding:18px'> <div style='display:flex;align-items:center;gap:12px'><div class='animal-card animal-turtle'>📊</div><div><div style='font-size:12px;color:#7f9aff'>Credit Score</div><div style='font-size:32px;font-weight:700'>{credit_score}</div></div></div>", unsafe_allow_html=True)
        st.plotly_chart(build_gauge(credit_score / 900, 'Score Gauge', '#4efca1'), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass-panel' style='padding:18px'><div style='font-size:12px;color:#7f9aff'>Risk Level</div>", unsafe_allow_html=True)
        st.plotly_chart(build_bar_chart({'Debt Load': metrics['Debt Load'], 'Stability': metrics['Stability'], 'Approval': approval}), width='stretch')
        st.markdown(build_progress('Approval Confidence', approval / 100, '#39f2be'), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='glass-panel' style='padding:18px'><div style='font-size:12px;color:#7f9aff'>Loan Approval Probability</div><div style='font-size:32px;font-weight:700'>{}%</div>".format(approval), unsafe_allow_html=True)
        st.plotly_chart(build_gauge(approval / 100, 'Approval Gauge', '#7f8dff'), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    # charts
    st.markdown('<div style="display:flex;gap:18px;margin-top:18px">', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel" style="flex:1">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0">Risk Drivers</h3>', unsafe_allow_html=True)
    st.plotly_chart(build_bar_chart(breakdown_norm), width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel" style="width:360px">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0">Risk Distribution</h3>', unsafe_allow_html=True)
    st.plotly_chart(build_pie_chart(breakdown_norm), width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # insights
    st.markdown('<div class="glass-panel" style="margin-top:16px">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0">Borrower insights</h3>', unsafe_allow_html=True)
    st.markdown(f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px'><div class='mini-card'><div class='mini-title'>Age</div><div class='metric-value'>{age} yrs</div></div><div class='mini-card'><div class='mini-title'>Income</div><div class='metric-value'>₹{income:,}</div></div><div class='mini-card'><div class='mini-title'>Loan (adj)</div><div class='metric-value'>₹{loan_adj:,}</div></div><div class='mini-card'><div class='mini-title'>Credit Score</div><div class='metric-value'>{credit_score}</div></div><div class='mini-card'><div class='mini-title'>Employment</div><div class='metric-value'>{employment_type}</div></div><div class='mini-card'><div class='mini-title'>Existing Loans</div><div class='metric-value'>{existing_count}</div></div></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="glass-panel" style="text-align:center;padding:36px;margin-top:18px"><h2>Ready to analyze credit risk</h2><p style="color:#9bb2ff">Use the sidebar to enter your details and see live predictions.</p></div>', unsafe_allow_html=True)

# small floating confetti (visual only)
money_html = ''
for i in range(10):
    left = random.randint(2, 96)
    delay = round(random.random() * 3, 2)
    dur = round(4 + random.random() * 3, 2)
    symbol = random.choice(['💵', '💸', '🪙'])
    money_html += f"<div style='position:fixed;left:{left}%;top:-10%;font-size:18px;opacity:0.8;animation:fall {dur}s linear {delay}s infinite'>{symbol}</div>"
st.markdown(f"<style>@keyframes fall{{0%{{transform:translateY(-15vh) rotate(0deg);opacity:0}}10%{{opacity:1}}100%{{transform:translateY(110vh) rotate(360deg);opacity:0}}}}</style><div style='pointer-events:none'>{money_html}</div>", unsafe_allow_html=True)
