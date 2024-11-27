import streamlit as st

testval = "hello"
testval = testval + "python"
print(testval)

st.title("UI test")
st.text("it is plain text")

if st.button("test Click") :
    st.text("u click button")
    
agree = st.checkbox("I agree?")
if agree == True :
    st.write("im agree")
else :
    st.write("im not agree")
    
onoff = st.toggle("toggle me")

if onoff :
    st.write("on")
else :
    st.write("off")
    
_radio = st.radio("radio",("A Btn","B btn","C btn"))
st.divider()

check1 = st.checkbox("checkbtn1")
check2 = st.checkbox("checkbtn2")
st.divider()

age = st.slider("how old r u",0,130,23)
st.write(f"your age is {age}")

if st.button("ok") :
    print(f"check box value : {check1}, {check2}")
    print(f"radion button value : {_radio}")
    print(f"slider bar value : {age}")