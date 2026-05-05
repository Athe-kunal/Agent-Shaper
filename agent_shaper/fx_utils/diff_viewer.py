import difflib
import streamlit as st

st.set_page_config(page_title="LLM Annotation Diff", layout="wide")
st.title("LLM Annotation Diff Viewer")

col_orig, col_llm = st.columns(2)
original_path = col_orig.text_input("Original file path")
rewritten_path = col_llm.text_input("LLM-annotated file path")

if original_path and rewritten_path:
    try:
        original = open(original_path).read()
        rewritten = open(rewritten_path).read()
    except OSError as e:
        st.error(str(e))
        st.stop()

    diff = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        rewritten.splitlines(keepends=True),
        fromfile="original",
        tofile="llm",
    ))

    if not diff:
        st.info("No differences found.")
    else:
        st.code("".join(diff), language="diff")
