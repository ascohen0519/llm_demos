import time
import os
import joblib
import streamlit as st
import google.generativeai as genai

st.title("AI Chatbot")

if 'can_run' not in st.session_state:
    st.session_state['can_run'] = False

if not st.session_state['can_run']:
  st.markdown('Please input a valid API key on the Demo Overview page')
  st.stop()

col1, col2 = st.columns(2)

ai_avatar = col1.selectbox('What would you like your AI\'s avatar to be?',
                              ["", "🔆","🌟","✨","🌈","🚀","🎉","🏆","🎱"],
                              help='Please select a character to use for your models responses')
user_avatar = col2.selectbox('What would you like your avatar to be?',
                                ["", "🐀","🐁","🐋","🐉","🐊","🐉","🐈","🐇","🐌","🐍","🐎","🐑","🐒","🐓","🐔","🐕","🐟","🐞","🐝","🐜","🐞","🐝","🐜","🐛","🐚","🐙","🐘","🐗","🐖","🐠","🐡","🐢","🐣","🐤","🐥","🐦","🐧","🐨","🐩","🐳","🐲","🐱","🐰","🐯","🐮","🐭","🐬","🐫","🐪","🐴","🐵","🐶","🐷","🐸","🐹","🐺","🐻","🐼"],
                                help='Please select a character to use for your responses')

if ai_avatar != '' and user_avatar != '':

    try:
        os.mkdir('data/')
    except:
        pass

    st.write('### Chat with AI✨ ')

    if 'ran' not in st.session_state:
        st.session_state.ran = False

    if not st.session_state.ran:
        st.session_state.messages = []
        st.session_state.gemini_history = []
        st.session_state.ran = True
        print('new_cache made')

    st.session_state.model = genai.GenerativeModel('gemini-pro')
    st.session_state.chat = st.session_state.model.start_chat(
        history=st.session_state.gemini_history,
    )


    for message in st.session_state.messages:
        with st.chat_message(
            name=message['role'],
            avatar=message.get('avatar'),
        ):
            st.markdown(message['content'])

    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = False
    if 'chat_title' not in st.session_state:
        st.session_state.chat_title = False


    if prompt := st.chat_input('Your message here...'):

        with st.chat_message('user', avatar=user_avatar):
            st.markdown(prompt)

        st.session_state.messages.append(
            dict(
                role='user',
                content=prompt,
                avatar=user_avatar
            )
        )

        response = st.session_state.chat.send_message(
            prompt,
            stream=True,
        )

        with st.chat_message(
            name='ai',
            avatar=ai_avatar,
        ):
            message_placeholder = st.empty()
            full_response = ''
            assistant_response = response

            for chunk in response:

                for ch in chunk.text.split(' '):
                    full_response += ch + ' '
                    time.sleep(0.05)

                    message_placeholder.write(full_response + '▌')

            message_placeholder.write(full_response)


        st.session_state.messages.append(
            dict(
                role='ai',
                content=st.session_state.chat.history[-1].parts[0].text,
                avatar=ai_avatar,
            )
        )
        st.session_state.gemini_history = st.session_state.chat.history

        joblib.dump(
            st.session_state.messages,
            f'data/{st.session_state.chat_id}-st_messages',
        )
        joblib.dump(
            st.session_state.gemini_history,
            f'data/{st.session_state.chat_id}-gemini_messages',
        )