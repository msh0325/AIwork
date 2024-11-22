import streamlit as st

#문장 받기
prompt = st.chat_input("궁금한것 물어보세용")

#추가적인 프롬프트
editor_text = st.text_area("추가적인 프롬프트에용",key="editor1",value="default text")

#prompt에 문장이 들어왔을 때 무언가 하기
if prompt :
    #콘솔에 프린트
    print(prompt)
    
    #사이트에 프린트
    st.text(f'당신은 방금 <{prompt}> 라고 말했어용.')
