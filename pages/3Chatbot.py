import time
import os
import joblib
import streamlit as st
import google.generativeai as genai

MODEL_ROLE = 'ai'

col1, col2 = st.columns(2)

AI_AVATAR_ICON = col1.selectbox('What would you like your robots avatar to be?',
                              ["", "🔆","🌟","✨","🌈","🚀","🎉","🏆","🎱"],
                              help='Cmon choose')
USER_AVATAR_ICON = col2.selectbox('What would you like your avatar to be?',
                                ["", "🐀","🐁","🐋","🐉","🐊","🐉","🐈","🐇","🐌","🐍","🐎","🐑","🐒","🐓","🐔","🐕","🐟","🐞","🐝","🐜","🐞","🐝","🐜","🐛","🐚","🐙","🐘","🐗","🐖","🐠","🐡","🐢","🐣","🐤","🐥","🐦","🐧","🐨","🐩","🐳","🐲","🐱","🐰","🐯","🐮","🐭","🐬","🐫","🐪","🐴","🐵","🐶","🐷","🐸","🐹","🐺","🐻","🐼"],
                                help='Cmon choose')

if AI_AVATAR_ICON != '' and USER_AVATAR_ICON != '':


    try:
        os.mkdir('data/')
    except:
        pass

    st.write('# Chat with AI✨ ')

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

        with st.chat_message('user', avatar=USER_AVATAR_ICON):
            st.markdown(prompt)

        st.session_state.messages.append(
            dict(
                role='user',
                content=prompt,
            )
        )

        response = st.session_state.chat.send_message(
            prompt,
            stream=True,
        )

        with st.chat_message(
            name=MODEL_ROLE,
            avatar=AI_AVATAR_ICON,
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
                role=MODEL_ROLE,
                content=st.session_state.chat.history[-1].parts[0].text,
                avatar=AI_AVATAR_ICON,
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