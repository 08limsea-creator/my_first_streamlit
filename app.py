import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="나의 첫 웹앱", page_icon="🎈")

# 제목과 설명
st.title("🎈 나의 첫 Streamlit 앱")
st.write("파이썬 코드 몇 줄로 만든 진짜 웹 앱입니다!")

# 사용자 입력 받기
name = st.text_input("이름을 입력하세요")

if name:
    st.success(f"{name}님, 환영합니다! 🎉")

# 슬라이더로 숫자 입력 받기
age = st.slider("나이를 선택하세요", 1, 100, 25)
st.write(f"선택한 나이: **{age}세**")

# 버튼 만들기
if st.button("풍선 날리기"):
    st.balloons()