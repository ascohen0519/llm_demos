import streamlit as st
import pdfplumber
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import numpy as np
import regex as re

st.title('Document Summarizer')


def click_button(button):
    """Sets session state button variable to True after user clicks button.

    Args:
        button: st.session_state variable.
    """

    st.session_state[button] = True


def construct_prompt(text_to_summarize, bullets_or_summary, max_bullets, paragraph_length):
    """Constructs summarization prompt from prompt template.

    Args:
        text_to_summarize: Text string of document provided by user.
        bullets_or_summary: User selected summary format.
        max_bullets: If user selects format with bullets, # of bullets selected.
        paragraph_length: If user selects format with paragraph, approximate word length selected.

    Returns:
        Final prompt to pass into model for summarization.
    """

    bullets_limit = 'please include %s bullets' % max_bullets

    paragraph_limit = 'the paragraph should be approximately %s words' % paragraph_length

    if bullets_or_summary == 'Bullets':
        summary_format = 'bullet'
        limits = bullets_limit

    elif bullets_or_summary == 'Paragraph':
        summary_format = 'a single paragraph'
        limits = paragraph_limit

    elif bullets_or_summary == 'Bullets & Paragraph':
        summary_format = 'a single paragraph and bullets'
        limits = paragraph_limit + ', and ' + bullets_limit

    else:
        return 'error'

    summarization_prompt = 'The following is a document:'
    summarization_prompt += '\nDOCUMENT: %s' % text_to_summarize
    summarization_prompt += '\nINSTRUCTIONS: Please write your summary in %s format. %s' % (summary_format, limits)
    summarization_prompt += '\nSUMMARY:'

    return summarization_prompt


def produce_summary(summarization_prompt, gen_config):
    """ Produces summary based on full summarization prompt and generation config determined by user selection.

    Args:
        prompt: Model prompt including document text to summarize and summarization instructions based on user input.
        gen_config: Dictionary of model configuration parameters based on user selection(s). Contains max_output_tokens
          by default. If user selects token algorithm other than greedy, contains explicit temperature setting along
          with top-p or top-k value.
    Returns:
        Model completion (summary).
    """

    return model.generate_content(
        contents=summarization_prompt,
        generation_config=genai.GenerationConfig(**gen_config),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
        }
    ).text


# Begin UI workflow for file/text input, summarization settings and displaying output.

# Redirect user to Demo Overview page for API key if no valid API key found.
if 'can_run' not in st.session_state:
    st.session_state['can_run'] = False

if not st.session_state['can_run']:
    st.markdown('Please input a valid API key on the Demo Overview page')
    st.stop()

# Configure Gemini 1.5 Flash model, instruct to be document summarizer.
genai.configure(api_key=st.session_state['gemini_api_key'])
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    system_instruction=['You are a helpful document summarizer.'])

# Prompt user to upload text for summarization via PDF import or manual text upload.
st.write('''
            ### Upload Text
            Upload your text via PDF or manual entry
            ''')

document_type = st.selectbox('What type of document would you like to summarize?',
                             ['', 'PDF', 'Manually input text'],
                             help='You can either upload a PDF document, or manually copy & paste text to summarize')

if document_type != '':
    text_to_summarize = ''

    # Convert PDF into single string of text.
    if document_type == 'PDF':
        uploaded_file = st.file_uploader('Choose your .pdf file', type='pdf')
        if uploaded_file:
            st.success('Successfully uploaded the file')
            with pdfplumber.open(uploaded_file) as file:
                all_pages = file.pages
                for i in all_pages:
                    text_to_summarize += i.extract_text()
    else:
        out = st.text_area('Enter your text, then click Upload', '', height=200)

        if 'text_entered' not in st.session_state:
            st.session_state.text_entered = False

        st.button('Upload', on_click=click_button, args=('text_entered',))

        if st.session_state.text_entered:
            text_to_summarize += out

    if len(text_to_summarize) > 0:
        num_characters = len(text_to_summarize)
        num_tokens = model.count_tokens(text_to_summarize).total_tokens
        num_words = len(re.findall('[a-zA-Z_]+', text_to_summarize))
        st.markdown(
            '''
            <span style='font-size: 12px; color:#0DA0D8;'>
            The length of your document is: \t %s characters, %s tokens,  %s words
            ''' % ('{:,}'.format(num_characters), '{:,}'.format(num_tokens), '{:,}'.format(num_words)),
            unsafe_allow_html=True)

        # Ensures text within gemini 1.5 flash context window length, accounting for 100 char of prompt instructions.
        # https://ai.google.dev/gemini-api/docs/long-context
        if num_tokens > 999900:
            st.write('this document is too long, please upload another document.')
            st.stop()

        # Begin user input summarization parameters.
        st.write('''
                ### Set Summarization Parameters
                Choose the format and length for your summary. Set optional advanced parameters.
                ''')

        bullets_or_summary = st.selectbox('What format would you like the summary in?',
                                          ['', 'Bullets', 'Paragraph', 'Bullets & Paragraph'])

        if 'summarize' not in st.session_state:
            st.session_state.summarize = False

        if bullets_or_summary != '':

            max_bullets, paragraph_length = 0, 0

            if bullets_or_summary == 'Bullets':
                max_bullets = st.select_slider('How many bullets for the bullet list?',
                                               options=[''] + list(range(2, 16)))

            elif bullets_or_summary == 'Paragraph':
                paragraph_length = st.select_slider('How many words for the paragraph (approx)?',
                                                    options=[''] + list(range(50, 1001, 50)))

            else:
                max_bullets = st.select_slider('How many bullets for the bullet list?',
                                               options=[''] + list(range(2, 16)))
                paragraph_length = st.select_slider('How many words for the paragraph (approx)?',
                                                    options=[''] + list(range(50, 1001, 50)))

            if (
                    (bullets_or_summary == 'Bullets' and max_bullets != '') or
                    (bullets_or_summary == 'Paragraph' and paragraph_length != '') or
                    (bullets_or_summary == 'Bullets & Paragraph' and max_bullets != '' and paragraph_length != '')):

                # If bullet output in format selection, assumes 100 max tokens per bullet.
                # If paragraph output in format selection, assumes 100 tokens per 75 words, plus a 20% ceiling buffer.
                default_bullet_tokens = max_bullets * 100
                default_paragraph_tokens = paragraph_length * (100 / 75) * 1.1
                default_max_tokens = round(int(default_bullet_tokens + default_paragraph_tokens), -2)

                # Default gen config includes max output tokens, calculated automatically based on prior selection.
                gen_config = {'max_output_tokens': default_max_tokens}

                # Optional input for advanced parameters (max tokens, token sampling, temperature).
                with st.expander('Advanced Parameters (optional)'):

                    # Option to increase max tokens.
                    final_max_tokens = st.select_slider('What is the max number of tokens for the entire summary?',
                                                        value=default_max_tokens,
                                                        options=[''] + list(
                                                            range(
                                                                default_max_tokens,
                                                                int(default_max_tokens * 1.2) + 1,
                                                                50)))

                    # Default to top-p token selection
                    gen_config = {'max_output_tokens': final_max_tokens}

                    # Option for token sampling method.
                    token_choice_algo = st.selectbox('Which method would you like to use for token selection?',
                                                     ['Top-p', 'Top-k', 'Greedy'])

                    if token_choice_algo == 'Top-p':

                        top_p = st.select_slider('What cumulative probability cutoff would you like to use?',
                                                 value=.95,
                                                 options=[
                                                     np.round(i * .01, 2) for i in list(range(1, 101, 1))])

                        gen_config['top_p'] = top_p

                        final_temp = st.select_slider('What temperature setting you would like to use?',
                                                      value=1,
                                                      options=[i * .01 for i in list(range(0, 175, 25))])

                        gen_config['temperature'] = final_temp

                    elif token_choice_algo == 'Top-k':

                        top_k = st.select_slider('How many top tokens to choose from?',
                                                 value=40,
                                                 options=list(range(2, 41, 1)))

                        gen_config['top_k'] = top_k

                        final_temp = st.select_slider('What temperature setting you would like to use?',
                                                      value=1,
                                                      options=[i * .01 for i in list(range(0, 175, 25))])

                        gen_config['temperature'] = final_temp

                    else:

                        gen_config = {'max_output_tokens': final_max_tokens}

                    # Dynamically display generation config. Display help link to educate user on parameter definitions.
                    st.write('current generation config:')
                    st.write(gen_config)
                    st.markdown(
                        "<span style='font-size: 12px;'>[learn more](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/adjust-parameter-values)</span>",
                        unsafe_allow_html=True)

                # Prompt for user to generate summary with current parameters.
                st.write('### Generate Summary')
                st.markdown('Click Summarize to Continue...')
                st.button('Summarize', on_click=click_button, args=('summarize',))

                if st.session_state.summarize:

                    try:
                        # Generate summary and display.
                        prompt = construct_prompt(text_to_summarize, bullets_or_summary, max_bullets, paragraph_length)
                        summary = produce_summary(prompt, gen_config)
                        st.write('# Summary:')
                        st.write('')
                        st.markdown(summary)
                        st.session_state.summarize = False
                    except:
                        # TODO: Add explicit error handling.
                        st.write('Error, please try again')
