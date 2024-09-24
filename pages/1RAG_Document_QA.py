import streamlit as st
import google.generativeai as genai
import pdfplumber
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np
import regex as re
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from google.oauth2 import service_account

st.title('RAG Document Q&A')

vertexai.init(
    project='gen-lang-client-0350351837',
    location='us-central1',
    credentials=service_account.Credentials.from_service_account_info(st.secrets["gcs_connections"]))


def click_button(button):
    """Sets session state button variable to True after user clicks button.

    Args:
        button: st.session_state variable.
    """

    st.session_state[button] = True


def update_question(question):
    """Sets session state question variable to the latest question entered by the user.

    Args:
        question: user entered question.
    """

    st.session_state.question = question


def get_chunks(chunk_approach, text, chunk_size, chunk_overlap):
    """Generates list of chunks from document, based on user inputs.

    Args:
        text: string of text from user input document.
        chunk_size: chunk size based on user input or default setting.
        chunk_overlap: chunk overlap based on user input or default setting.

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


def get_embedding(text, embedding_model_choice, dimensionality, retrieval_type):
    """Generates embedding for given text input.

    See https://ai.google.dev/gemini-api/docs/embeddings for more info.

    Args:
        text: string of text to embed.
        embedding_model_choice: gemini embedding model selected by user in advanced settings.
        dimensionality: # of dimensions for embedding vector, selected by user in advanced settings.
        retrieval_type: RETRIEVAL_QUERY for user query embedding, RETRIEVAL_DOCUMENT for chunk embeddings.

    Returns:
        Embedding vector values.
    """

    model = TextEmbeddingModel.from_pretrained(model_name=embedding_model_choice)
    text_input = [TextEmbeddingInput(text=text, task_type=retrieval_type)]
    return model.get_embeddings(texts=text_input, output_dimensionality=dimensionality)[0].values


def find_best_chunk(question, embedding_model_choice, dimensionality, dataframe, num_top_chunks):
    """Generates text embedding for user question and finds nearest chunk embedding from dataframe.

    Args:
        question: user entered question.
        embedding_model_choice: gemini embedding model selected by user in advanced settings.
        dimensionality: # of dimensions for embedding vector, selected by user in advanced settings.
        dataframe: documents vector database in pandas dataframe.
        num_top_chunks: number of chunks to pass to LLM as context in final prompt.

    Returns:
        Text string of most similar chunk(s) from dataframe to feed into prompt with question.
    """

    query_embedding = get_embedding(
        text=question,
        embedding_model_choice=embedding_model_choice,
        dimensionality=dimensionality,
        retrieval_type='RETRIEVAL_QUERY')

    dataframe['dot_products'] = np.dot(np.stack(dataframe['embedding']), query_embedding)

    idx = dataframe.sort_values('dot_products', ascending=False).reset_index()

    # Maximum number of chunks is total available chunks from document.
    num_top_chunks_final = np.min([num_top_chunks, len(dataframe)])

    return [idx.iloc[n].text for n in range(num_top_chunks_final)]


def construct_prompt(document_chunk_list, question):
    """Constructs prompt to pass to model based on relevant identified document chunks and user question.

    Args:
        document_chunk_list: list of chunks identified as best chunks.
        question: user question.

    Returns:
        String of full prompt to pass into model.
    """

    qa_prompt = 'The following is a set of documents, and a question about the documents: \n'

    for i in range(len(document_chunk_list)):
        qa_prompt += '\nDOCUMENT NUMBER %s: %s. END DOCUMENT.\n' % (i + 1, document_chunk_list[i])

    qa_prompt += '\nQUESTION: %s \n' % question

    qa_prompt += ('\nINSTRUCTIONS: Please review documents in order and answer the users question as soon as you find' \
                  ' the answer. Your response should only include the answer to the question. If you are unable to' \
                  ' find the answer, respond "Information not found".\n')

    qa_prompt += '\nANSWER: '

    return qa_prompt


def produce_answer(prompt):
    """ Produces answer based on prompt containing ordered best document chunk(s) and user question.

    Args:
        prompt: Final prompt to pass into LLM for answer, including user question and relevant document chunk(s).

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


chunk_method_default = 'Fixed size chunking'
chunk_size_default = 1024
chunk_overlap_default = 20
num_top_chunks_default = 3
embedding_model_choice_default = 'text-embedding-004'
dimensionality_default = 768

if not 'df' in st.session_state:
    st.session_state['df'] = pd.DataFrame()

if 'question' not in st.session_state:
    st.session_state.question = ''

if 'answer' not in st.session_state:
    st.session_state.answer = ''

# Begin UI workflow for file/text input, document chunk settings.

# Redirect user to Demo Overview page for API key if no valid API key found.
if 'can_run' not in st.session_state:
    st.session_state['can_run'] = False

if not st.session_state['can_run']:
    st.markdown('Please input a valid API key on the Demo Overview page')
    st.stop()

# Configure Gemini 1.5 Flash model, instruct to be document Q&A assistant.
genai.configure(api_key=st.session_state['gemini_api_key'])
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    system_instruction=['You are a helpful document question-answering service.'])

# Prompt user to upload text for summarization via PDF import or manual text upload.
st.write('''
        ### Upload Text
        Upload your text via PDF or manual entry
        ''')

document_type = st.selectbox('What type of document would you like to ask questions about?',
                             ['', 'PDF', 'Manually input text'],
                             help='You can either upload a PDF document, or manually copy & paste text to summarize')

if document_type != '':
    text_for_qa = ''

    # Convert PDF into single string of text.
    if document_type == 'PDF':
        uploaded_file = st.file_uploader('Choose your .pdf file', type='pdf')
        if uploaded_file:
            st.success('Successfully uploaded the file')
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

    # Display character, token and word count.
    if len(text_for_qa) > 0:
        num_characters = len(text_for_qa)
        num_tokens = model.count_tokens(text_for_qa).total_tokens
        num_words = len(re.findall('[a-zA-Z_]+', text_for_qa))
        st.markdown(
            '''
            <span style='font-size: 12px; color:#0DA0D8;'>
            The length of your document is: \t %s characters, %s tokens,  %s words
            ''' % ('{:,}'.format(num_characters), '{:,}'.format(num_tokens), '{:,}'.format(num_words)),
            unsafe_allow_html=True)

        # Begin user input chunking parameters.
        st.write('''
                ### Set Q&A Parameters
                Set optional advanced parameters.
                ''')

        d_model = {
            'Auto': CharacterTextSplitter,
            'Fixed size chunking': CharacterTextSplitter,
            'Recursive chunking': RecursiveCharacterTextSplitter}

        customize_setting = st.selectbox('How would you like to set chunking and embedding parameters?',
                                         options=[
                                             '',
                                             'Auto',
                                             'Customized'],
                                         help='''
                                            This determines the logic and parameters used to break your document into
                                            chunks, and the model and parameters used for document and query embeddings.
                                            ''')

        if customize_setting != '':

            if customize_setting == 'Auto':
                chunk_method = chunk_method_default
                chunk_size = chunk_size_default
                chunk_overlap = chunk_overlap_default
                num_top_chunks = num_top_chunks_default
                embedding_model_choice = embedding_model_choice_default
                dimensionality = dimensionality_default

            else:

                chunk_method = st.selectbox('How would you like to process the document into chunks',
                                            options=['Fixed size chunking', 'Recursive chunking'])
                chunk_size = st.select_slider('How many characters per chunk?',
                                              options=list(range(1, 2049, 1)),
                                              value=chunk_size_default)
                chunk_overlap = st.select_slider('How many characters to overlap between chunks?',
                                                 options=list(range(0, 301, 10)),
                                                 value=chunk_overlap_default)
                num_top_chunks = st.select_slider('How many chunks to retrieve for LLM in final prompt?',
                                                  options=list(range(1, 11, 1)),
                                                  value=num_top_chunks_default)

                embedding_model_choice = st.selectbox(
                    'Choose your embedding model',
                    options=['text-embedding-004', 'text-multilingual-embedding-002', 'text-embedding-preview-0815'])

                dimensionality = st.select_slider('What dimensionality would you like to use for embeddings?',
                                                  options=list(range(1, 769)),
                                                  value=dimensionality_default)

            if 'settings_ready' not in st.session_state:
                st.session_state.settings_ready = False

            st.button('Generate Chunks & Embeddings', on_click=click_button, args=('settings_ready',))

            if st.session_state.settings_ready:
                # Generate chunks and embedding vectors for each chunk
                if (chunk_size != ''
                    and chunk_overlap != ''
                    and num_top_chunks != ''
                    and embedding_model_choice != ''
                    and dimensionality != ''):

                    st.write('generating chunks...')
                    chunks = get_chunks(d_model[chunk_method], text_for_qa, chunk_size, chunk_overlap)

                    st.write('generating embeddings (this may take a minute)...')

                    st.session_state.df = pd.DataFrame(chunks, columns=['text'])
                    st.session_state.df['embedding'] = st.session_state.df.apply(
                        lambda x: get_embedding(
                            text=x.text,
                            embedding_model_choice=embedding_model_choice,
                            dimensionality=dimensionality,
                            retrieval_type='RETRIEVAL_DOCUMENT'),
                        axis=1)
                    st.session_state.embedded = True

                    # For user input question, find most relevant chunk(s), construct final LLM prompt, yield completion.
                    st.session_state.question = st.text_input(
                        'Ask me a question about the contents of your document:')

                    if st.session_state.question != '':
                        document_chunk_list = find_best_chunk(
                            st.session_state.question, embedding_model_choice, dimensionality, st.session_state.df,
                            num_top_chunks)

                        prompt = construct_prompt(document_chunk_list, st.session_state.question)
                        st.session_state.answer = produce_answer(prompt)

                        st.markdown(st.session_state.answer)

                        st.session_state.answer = ''
                        st.session_state.question = ''

                        # Debugging shows summary of chunks created, relevant chunks identified, and displays final prompt.
                        with st.expander('debug'):
                            st.write('Embedding model choice: %s' % embedding_model_choice)
                            st.write('Embedding dimensionality: %s' % str(dimensionality))
                            st.write('Chunk size setting: %s' % chunk_size)
                            st.write('Chunk overlap setting: %s' % chunk_overlap)
                            st.write('Number of chunks generated: %s' % len(chunks))
                            st.write(
                                'Avg character length of generated chunk: %s' % round(num_characters / len(chunks), 1))
                            st.write('first, second and last embedding:')
                            st.write(pd.concat([st.session_state.df.head(2), st.session_state.df.tail(1)]))
                            st.write('Final prompt:')
                            st.write(prompt)
                else:
                    st.write('try again')
