# session
import streamlit as st

if 'counter' not in st.session_state :
    st.session_state.counter = 0

#counter =0 이걸 사용하면 버튼을 누를 때마다 다시 실행되서 1 이상으로 늘어나지 않음. 세션방식으로 저장?해야함

if st.button("increase") :
    #counter = counter+1
    st.session_state.counter +=1
    
st.write(f"counter : {st.session_state.counter}")