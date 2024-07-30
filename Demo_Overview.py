import streamlit as st
import google.generativeai as genai
import time

st.title('💬 LLM App Demos, by Aaron Cohen')
st.caption('[source code](https://github.com/ascohen0519/llm_demos/tree/main/pages)')

'### Instructions'
'''
In order to run the demos, you'll need to input your API key:
'''
if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = False

st.session_state['gemini_api_key'] = st.text_input('Gemini API Key')

if not st.session_state['gemini_api_key']:
    st.info('''
    Please add your Gemini API Key to Continue.
    If you don\'t have one, you can get an API key [here](https://aistudio.google.com/app/apikey).
    ''')
    st.stop()
else:
    genai.configure(api_key=st.session_state['gemini_api_key'])
    model = genai.GenerativeModel('gemini-1.5-flash')

try:
    model.generate_content('hello')
    st.write('Valid API key, thank you.')
    st.session_state['can_run'] = True
except:
    ('This is not a valid API key, please try again')
    st.stop()

'#### Please choose a demo from the sidebar on the left:'

'## Document Summarizer'

'''
Upload PDFs or manually copy and paste a body of text to be summarized. You can set parameters
for the summary to be generated, including format, length, and advanced parameters including token sampling method
and temperature.
'''

'## Chatbot'

'''
Chat with AI about a topic of interest. The model will remember your conversation history throughout the session. You 
can customize your AI's name, avatar and desired creativity level. 
'''

'## Document Q&A'
st.write('(testing in progress)')

'''
Upload PDFs or manually copy and paste a body of text. After a brief period, you'll be able to ask questions
about your upload and receive answers. You can also set advanced parameters, including chunk
size, chunk overlap, and how many potential answer results you would like to see (in progress).
'''