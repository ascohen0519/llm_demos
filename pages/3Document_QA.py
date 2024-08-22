import streamlit as st
import google.generativeai as genai
import pdfplumber
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
import time
import numpy as np
import regex as re

st.title('Document Q&A')
st.markdown(
            '''
            <span style='font-size: 12px; color:#cd6155;'>
            This app is currently in testing and provides basic retrieval functionality only.<br>
            Upcoming changes will include optimal N chunks to pass into prompt and document specific settings for
            default chunk size.
            ''',
            unsafe_allow_html=True)
st.write('\n')

def click_button(button):
    """Sets session state button variable to True after user clicks button.

    Args:
        button: st.session_state variable.
    """
    st.session_state[button] = True

def update_question(question):
    """Sets st.session_state.question variable to the latest question entered by the user.

    Args:
        question: user entered question.
    """
    st.session_state.question = question

def get_chunks(chunk_approach, text, chunk_size, chunk_overlap):
    """Generates list of chunks from text string based on user inputs.

    Args:
        text: text corresponding to user entered document.
        chunk_size: chunk size based on user manual input.
        chunk_overlap: chunk overlap based on user manual input.

    Returns:
        List of document chunks.
    """

    kwargs = {'chunk_size': chunk_size, 'chunk_overlap': chunk_overlap}

    if chunk_approach == CharacterTextSplitter:
        kwargs['separator'] = ''
    else:
        pass
    text_splitter = chunk_approach(**kwargs)

    return text_splitter.split_text(text)

def get_embedding(text):
    """Generates text embedding for given text input.

    See https://ai.google.dev/gemini-api/docs/embeddings for more info.

    Args:
        text: text corresponding to user entered question.

    Returns:
        Embedding vector.
    """

    return genai.embed_content(model='models/text-embedding-004',
                               content=text,
                               task_type='retrieval_document')['embedding']

def build_prompt(document_chunk_list, question):
    """Generates prompt to pass to model based on relevant identified document chunks and user question.

    Args:
        document_chunk_list: list of chunks identified as best chunks from document based on user input settings.
        question: user question.

    Returns:
        String of full prompt to pass into model.
    """

    qa_prompt = 'The following is a set of documents, and a question about the documents: \n'

    for i in range(len(document_chunk_list)):
        qa_prompt += '\nDOCUMENT NUMBER %s: %s. END DOCUMENT.\n'%(i+1, document_chunk_list[i])

    qa_prompt += '\nQUESTION: %s \n'%question

    qa_prompt += ('\nINSTRUCTIONS: Please review documents in order and answer the users question as soon as you find' \
                  ' the answer. Your response should only include the answer to the question. If you are unable to' \
                  ' find the answer, respond "Information not found".\n')

    qa_prompt += '\nANSWER: '

    return qa_prompt


def find_best_chunk(question, dataframe, num_top_chunks):
    """Generates text embedding for user question and finds nearest chunk embedding from dataframe.

    Args:
        question: user entered question
        dataframe: documents vector database in pandas dataframe

    Returns:
        Text string of most similar chunk from dataframe to feed into prompt with question.
    """
    query_embedding = get_embedding(question)
    dataframe['dot_products'] = np.dot(np.stack(dataframe['embedding']), query_embedding)
    idx = dataframe.sort_values('dot_products', ascending=False).reset_index()

    num_top_chunks_final = np.min([num_top_chunks, len(dataframe)])

    return [idx.iloc[n].text for n in range(num_top_chunks_final)] # dataframe.iloc[idx]['text']

def produce_answer(prompt):
    """ Produces answer based on prompt input, containing best document chunk and user question.

    Args:
        document_chunk_list: List of best document chunks for question from vector store.
        question: User question.

    Returns:
        Model completion (answer).
    """

    return model.generate_content(
        contents=prompt,
        generation_config=genai.GenerationConfig(temperature=1),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
        }
    ).text

# Begin UI workflow for file/text input, document chunk settings.

# Redirect user to Demo Overview page for API key if no valid API key found.
if 'can_run' not in st.session_state:
    st.session_state['can_run'] = False

if not st.session_state['can_run']:
    st.markdown('Please input a valid API key on the Demo Overview page')
    st.stop()

# Configure Gemini 1.5 Flash model, instruct to be document Q&A assistant.
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#supported_models
genai.configure(api_key=st.session_state['gemini_api_key'])
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    system_instruction=[
        'You are a helpful document question-answering service.'])

# Prompt user to upload text for summarization via PDF import or manual text upload.
st.write('''
        ### Upload Text
        Upload your text via pdf or manual entry
        ''')

document_type = st.selectbox('What type of document would you like to ask questions about?',
                             ['', 'Manually input text', 'PDF'],
                             help='You can either upload a PDF document, or manually copy & paste text to summarize')

if document_type != '':
    text_for_qa = ''

    # Convert PDF into single string of text. If PDF is multiple pages, concatenate all pages via '. ' first.
    if document_type == 'PDF':
        uploaded_file = st.file_uploader('Choose your .pdf file', type='pdf')
        if uploaded_file:
            st.success('Uploaded the file')
            with pdfplumber.open(uploaded_file) as file:
                all_pages = file.pages
                for i in all_pages:
                    text_for_qa += i.extract_text()
    else:
        out = st.text_area('Enter your text, then click Upload', '', height=200)

        if 'text_entered' not in st.session_state:
            st.session_state.text_entered = False

        st.button('Upload', on_click=click_button, args=('text_entered',))

        if st.session_state.text_entered:
            text_for_qa += out

    if len(text_for_qa) > 0:
        num_characters = len(text_for_qa)
        num_tokens = model.count_tokens(text_for_qa).total_tokens
        num_words = len(re.findall('[a-zA-Z_]+', text_for_qa))
        st.markdown(
            '''
            <span style='font-size: 12px;'> 
            The length of your document is: \t %s characters, %s tokens,  %s words
            '''%('{:,}'.format(num_characters), '{:,}'.format(num_tokens), '{:,}'.format(num_words)),
            unsafe_allow_html=True)

        # Begin user input summarization parameters.
        st.write('''
                ### Set Q&A Parameters
                Set optional advanced parameters.
                ''')

        d_model = {
            'Auto': CharacterTextSplitter,
            'Fixed size chunking': CharacterTextSplitter,
            'Recursive chunking': RecursiveCharacterTextSplitter}
            # TODO: investigate Streamlit broken pipe error for llama-index package.
            #'Sentence chunking': SentenceSplitter}

        chunk_setting = st.selectbox('How would you like to process this document into chunks?',
                                       options=[
                                           '',
                                           'Auto',
                                           'Fixed size chunking',
                                           'Recursive chunking'],
                                           # 'Sentence chunking'],
                                        help='''
                                            This determines the logic and parameters used to break your document into
                                            chunks.
                                            ''')

        if chunk_setting != '':

            if chunk_setting == 'Auto':
                chunk_size = 1024
                chunk_overlap = 20
                num_top_chunks = 3

            else:
                chunk_size = st.select_slider('How many characters per chunk?',
                                               options=[''] + list(range(1, 2001, 1)))
                chunk_overlap = st.select_slider('How many characters to overlap between chunks?',
                                               options=[''] + list(range(0, 101, 1)))
                num_top_chunks = st.select_slider('How many chunks to retrieve for LLM in final prompt?',
                                               options=[''] + list(range(1, 10, 1)))

            if chunk_size != '' and chunk_overlap != '' and num_top_chunks != '':

                st.write('generating chunks...')
                chunks = get_chunks(d_model[chunk_setting], text_for_qa, chunk_size, chunk_overlap)

                st.write('generating embeddings (this may take a minute)...')

                if not 'df' in st.session_state:
                    st.session_state['df'] = pd.DataFrame()

                st.session_state.df = pd.DataFrame(chunks, columns=['text'])
                st.session_state.df['embedding'] = st.session_state.df.apply(lambda x: get_embedding(x.text), axis=1)
                st.session_state.embedded = True

                st.write('done, ready for Q&A')

                if 'question' not in st.session_state:
                    st.session_state.question = ''

                if 'answer' not in st.session_state:
                    st.session_state.answer = ''

                st.session_state.question = st.text_input(
                    'Ask me a question about the contents of your document:')

                if st.session_state.question != '':
                    with st.spinner('Wait for it...'):
                        time.sleep(2)

                    document_chunk_list = find_best_chunk(
                        st.session_state.question, st.session_state.df, num_top_chunks)

                    prompt = build_prompt(document_chunk_list, st.session_state.question)
                    st.session_state.answer = produce_answer(prompt)

                    st.markdown(st.session_state.answer)

                    st.session_state.answer = ''
                    st.session_state.question = ''

                    with st.expander('debug'):
                        st.write('Chunk size setting: %s'%chunk_size)
                        st.write('Chunk overlap setting: %s' % chunk_overlap)
                        st.write('Number of chunks generated: %s'%len(chunks))
                        st.write('Avg character length of generated chunk: %s' % round(num_characters / len(chunks), 1))
                        st.write('Best chunks identified for current question:')
                        st.write(document_chunk_list)
                        st.write('first, second and last chunk:')
                        st.write(chunks[:2] + chunks[-1:])
                        st.write('first, second and last embedding:')
                        st.write(pd.concat([st.session_state.df.head(2), st.session_state.df.tail(1)]))
                        st.write('Final prompt:')
                        st.write(prompt)