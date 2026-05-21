import py_compile,traceback
try:
    py_compile.compile(r"d:/CreditRiskAnalyzer/streamlit_app.py", doraise=True)
    print('Compiled OK')
except Exception:
    traceback.print_exc()
