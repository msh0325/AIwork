import streamlit as st

st.title("Side bar Sample")

with st.sidebar :
    st.title("Sidebar")
    _radio = st.radio(
        "choose method",
        ("Stnadard", "Express")
    )
    msg = st.text_input(
        label="enter msg",
        placeholder="input any message"
    )

if st.button("submit") :
    print(f"radio : {_radio}")
    print(f"message : {msg}")
    st.write(f"message : {msg}")
    st.write(f" radio : {_radio}")