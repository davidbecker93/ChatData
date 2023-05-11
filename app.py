import streamlit as st 
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import PandasAI
import openai

load_dotenv()

# store the API key in Secrets and add here
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat_with_csv(df,prompt):
    llm = OpenAI(api_token=openai.api_key)
    pandas_ai = PandasAI(llm)
    result = pandas_ai.run(df, prompt=prompt)
    print(result)
    return result

def translate_text(text):
    prompt = f"Translate the following English text to VietNamese: {text}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    translation = response.choices[0].message.content.strip()
    return translation

st.set_page_config(layout='wide')

st.title("Dữ liệu lịch mổ từ Bệnh viện")

#input_csv = st.file_uploader("Upload your CSV file", type=['csv'])
input_csv = ('lichmo.csv')

if input_csv is not None:

        col1, col2 = st.columns([1,1])

        with col1:
            st.info("Dữ liệu đã load thành công")
            data = pd.read_csv(input_csv)
            st.dataframe(data, use_container_width=True)

        with col2:

            st.info("Trò chuyện với dữ liệu")
            
            input_text = st.text_area("Đặt câu hỏi")

            if 'count' not in st.session_state:
                st.session_state.count = 0

            if input_text is not None and st.session_state.count < 5:
                if st.button("Chat với Data"):
                    st.info("Bạn: "+input_text)
                    result = chat_with_csv(data, input_text)
                    trans = translate_text(result)
                    st.success(trans)
                    st.session_state.count += 1
                    st.experimental_set_query_params(count=st.session_state.count)
            elif st.session_state.count >= 5:
                st.success("Bạn đã sử dụng hết số lần chat với Data.")
                st.experimental_set_query_params(count=st.session_state.count)

