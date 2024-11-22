import streamlit as st

#타이틀
st.title("My first App")
#텍스트
st.text("my hello my friends")
#강조문 *하나면 기울이기, 두개면 볼드체
st.markdown("*Streamlit* is **good**")
#latex 문법을 사용할 수 있음. 수식 넣기 가능하다는 뜻
st.latex("a+ar + a r^2 + a r^3 ") 
st.latex("\sum_{k=0}^{n-1} ar^k") #시그마도 가능!

#실행 명령어
#streamlit run <파일명>