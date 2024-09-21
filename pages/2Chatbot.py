import time
import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image

st.title('AI Chatbot')

def append_message(role, content):
    """Appends message to message history.

    Args:
        role: Model or user.
        content: Content in message (completion from model or prompt from user).
    """

    st.session_state.messages.append({'role': role, 'parts': [content]})


def give_response(message_history, gen_config):
    """Generates model response given message history and latest user prompt.

    Args:
        message_history: Session state message history, containing all prior messages and latest user prompt.
        gen_config: Dictionary of top-p and temperature setting based on user input creativity level.

    Returns:
        Model response, includes .text for displaying the prompt completion in UI.
    """

    return st.session_state.model.generate_content(
        contents=message_history,
        generation_config=genai.GenerationConfig(**gen_config),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
        }
    )


# If not valid API key entered, direct user to initial Demo Overview page.
if 'can_run' not in st.session_state:
    st.session_state['can_run'] = False

if not st.session_state['can_run']:
    st.markdown('Please input a valid API key on the Demo Overview page')
    st.stop()

# Company name and logo customization.
col1, col2 = st.columns(2)

default_ai_avatar = ''

company_name = col1.text_input('What is your companies name? (optional)', '')

company_logo = col2.file_uploader('Add your company logo (optional), file type .png', type='png')
css = '''
<style>
    [data-testid='stFileUploader'] {
        width: max-content;
    }
    [data-testid='stFileUploader'] section {
        padding: 0;
        float: left;
    }
    [data-testid='stFileUploader'] section > input + div {
        display: none;
    }
    [data-testid='stFileUploader'] section + div {
        float: right;
        padding-top: 0;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

if company_logo:
    company_logo_new = Image.open(company_logo)
    company_logo_resized = company_logo_new.resize((200, 200))
    default_ai_avatar = 'My company logo'
else:
    company_logo_resized = ''

# Chatbot name and avatar customization.
ai_avatar_list = ['🔆', '🌟', '✨', '🌈', '🚀', '🎉', '🏆', '🎱']

ai_avatar = col1.selectbox('What would you like your Chatbot\'s avatar to be?',
                           [default_ai_avatar] + ai_avatar_list)

if ai_avatar == 'My company logo':
    if company_logo:
        ai_avatar = company_logo_resized
    else:
        st.markdown(
            '''
            <span style='font-size: 12px; color:#C53F27;'>
            Please upload your company logo, or choose an avatar from the list
            ''', unsafe_allow_html=True)
else:
    pass

user_avatar = col2.selectbox('What would you like your avatar to be?',
                             ['', '🐬', '🐶', '🐉', '🐈', '🐙', '🐘', '🐹', '🐸'])

ai_name = col1.text_input('What would you like your Chatbot\'s name to be?', '')

# Model temperature / token selection method customization.
ai_creativity = col2.selectbox('How creative would you like your Chatbot to be?',
                               ['', 'Not creative at all', 'Moderately creative', 'Very creative'],
                               help='''
                                      This decides how your model chooses the next word in it's response to you, by
                                      altering the temperature and top-p configuration settings.
                                      ''')

d_avatar = {'model': ai_avatar, 'user': user_avatar}

# If all choices selected: define generation config, initiate model and open chat window.
if ai_avatar != '' and user_avatar != '' and ai_name != '' and ai_creativity != '':

    st.markdown("""<p style="height:2px;border:none;color:#AED6F1;background-color:#AED6F1;" /> """, unsafe_allow_html=True)

    # User entered creativity determines gen config temperature and top-p.
    d_creativity_temp = {'Not creative at all': 0, 'Moderately creative': .75, 'Very creative': 1.5}
    d_creativity_topp = {'Not creative at all': 0, 'Moderately creative': .5, 'Very creative': .95}
    gen_config = {'temperature': d_creativity_temp[ai_creativity], 'top_p': d_creativity_topp[ai_creativity]}

    # Set chatbot header to default or customized to Company.
    col1, col2 = st.columns([10, 1])

    if company_name:
        chatbot_name = company_name + ' Chatbot'
    else:
        chatbot_name = 'Chat with AI'
    col1.write('### %s'%(chatbot_name))

    if company_logo_resized != '':
        col2.image(company_logo_resized)
    else:
        pass

    # If first run, instantiate model and create empty list to append future prompts for ongoing context.
    if 'ran' not in st.session_state:
        st.session_state.ran = False

    if not st.session_state.ran:
        # Create message history.
        st.session_state.messages = []

        # Configure Gemini 1.5 Flash model, instruct to be general purpose chatbot.
        genai.configure(api_key=st.session_state['gemini_api_key'])
        st.session_state.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction='You are a general purpose chatbot. Your name is %s' % ai_name)

        # Display welcome message
        if company_name:
            ai_name_string = ai_name + ', a general purpose chatbot brought to you by ' + company_name
        else:
            ai_name_string = ai_name
        intro = 'Hello, my name is %s, how may I help you today?' % ai_name_string
        append_message('model', intro)
        st.session_state.ran = True

    # For all messages in history, re-write in chat ui.
    for message in st.session_state.messages:
        with st.chat_message(name=message['role'], avatar=d_avatar[message['role']]):
            st.markdown(message['parts'][0])

    # Once new prompt is sent, display user prompt, append to message history, and display model response in stream.
    if prompt := st.chat_input('Your message here...'):

        # Display user prompt, and append to message history.
        with st.chat_message('user', avatar=user_avatar):
            st.markdown(prompt)

        # Append latest user prompt to history.
        append_message('user', prompt)

        # Feed model entire history up to latest prompt, gen_config set based on user input creativity level.
        response = give_response(st.session_state.messages, gen_config)

        # Display model response in stream.
        with st.chat_message('model', avatar=ai_avatar):
            message_placeholder = st.empty()
            full_response = ''
            try:
                for chunk in response:
                    for ch in chunk.text.split(' '):
                        full_response += ch + ' '
                        time.sleep(0.05)
                        message_placeholder.write(full_response + '▌')
                message_placeholder.write(full_response)

                # Append model response to history.
                append_message('model', response.text)

            except:
                error_response = 'I can not answer that for you, please try another prompt. Thank you.'
                message_placeholder.write(error_response)

                # Append model response to history.
                append_message('model', error_response)
