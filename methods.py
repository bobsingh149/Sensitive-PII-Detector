import os

from spacy.tokens import DocBin
from tqdm import tqdm
from subprocess import check_output
import streamlit as st
from st_aggrid import AgGrid

import spacy
from spacy import displacy
from collections import defaultdict
from streamlit_card import card

import evaluate
from items import displacy_options
from nltk.stem import PorterStemmer
from wordcloud import STOPWORDS
import re
from streamlit_ace import st_ace
import stanza.utils.datasets.ner.prepare_ner_file as prepare_ner_file
import pandas as pd
from sklearn.model_selection import train_test_split
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, StackedEmbeddings
import spacy_stanza
from annotated_text import annotated_text
import plotly.express as px



keys = ["d0", "d1", "d2"]
data_key = keys[0]
name_key = 'name'
file_type_key = 'type'
model_name = 'model'
lib_key = 'lib'
size_key = 'size'

from evaluate import  get_metrics

stopwords = set(STOPWORDS)
ps = PorterStemmer()


def getdoc(text,gold=False):

    entities = []


    nlp = spacy.load("./en_core_web_sm-3.4.0")
    doc = nlp(text)


    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))

    return [(text, {'entities': entities})]


def preprocess(sentence):
    processed = []

    for word in sentence.split():
        word = str(word)

        word = re.sub(r'[^\w\s]', '', word)

        word = word.lower()
        word = ps.stem(word)

        if word not in stopwords:
            processed.append(word)

    return ' '.join(processed)


@st.cache(allow_output_mutation=True)
def get_data(file):
    data = pd.read_csv(file)
    return data


def space(n):
    for i in range(n):
        st.write('')


def local_html(file_name):
    with open(file_name) as f:
        st.markdown(f.read(), unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def read_txt(file, write=True):
    text = ''
    encoding = 'utf-8'

    for line in file:
        text += line.decode(encoding)

    if write:
        st.info(text[:3000])

    return text


def read_csv(file):
    data = get_data(file)
    AgGrid(data, theme='alpine', columns_auto_size_mode=True, height=500)
    return data


def read_tsv(file):
    data = pd.read_csv(file, sep="\t", header=None, encoding="latin1")
    data.columns = data.iloc[0]
    data = data[1:]
    data.columns = ['Index', 'Sentence', 'Word', 'POS', 'Tag']
    data = data.reset_index(drop=True)

    AgGrid(data, theme='alpine', columns_auto_size_mode=True, height=500)
    return data


def file_upload():
    files = st.file_uploader('Choose a file', type=['csv', 'xlsx', 'xls', 'txt', 'tsv'],
                             accept_multiple_files=True)

    # if 'files' not in st.session_state:
    #     st.session_state.files = files

    for idx, file in enumerate(files):

        st.write('Preview')

        file_type = file.name.partition('.')[-1]

        st.caption(f'.{file_type} file')

        if file_type == 'csv' or file_type == 'xls' or file_type == 'xlsx':
            data = read_csv(file)
        elif file_type == 'tsv':
            data = read_tsv(file)
        else:
            data = read_txt(file)

        key = "d" + str(idx)

        # print(key)

        st.session_state[data_key] = data
        st.session_state[name_key] = file.name
        st.session_state[file_type_key] = file_type

        st.markdown('---')

    if data_key in st.session_state and len(files) == 0:

        if (st.session_state[file_type_key] == 'txt'):
            st.success(st.session_state[data_key])

        else:
            space(1)
            AgGrid(st.session_state[data_key], theme='alpine', height=500, columns_auto_size_mode=True)


def process_text(corpus, size):
    dataset = []

    for line in corpus.split('\n')[:size]:
        text = line.rstrip()

        # dataset.append((preprocess(text)))
        dataset.append(text)

    return dataset


def process_df(df, size):
    df = df.head(size)

    dataset = []

    df.fillna('', inplace=True)

    np_arr = df.to_numpy()

    for i in range(len(np_arr)):
        text = ' '.join(str(v) for v in np_arr[i])

        dataset.append(preprocess(text))

    return dataset


def flair_train_model(data, size):
    st.info('Preparing the datasets')

    with open('mod_flair_config.txt', 'r') as f:
        config = f.read()

    config_list = config.split('\n')
    paras = {}

    for conf in config_list:

        key, info = conf.split(':')

        type, val = info.split('=')

        val = val[:-1]
        key = key.strip()
        type = type.strip()
        val = val.strip()


        if type == 'str':
            val = str(val)

        elif type == 'int':
            val = int(val)

        elif type == 'float':
            val = float(val)

        elif type == 'bool':

            if val == 'True':
                val = True
            else:
                val = False

        paras[key] = val



    class getsentence(object):

        def __init__(self, data):
            self.n_sent = 1.0
            self.data = data
            self.empty = False

            def agg_func(s): return [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                                  s["POS"].values.tolist(),
                                                                  s["Tag"].values.tolist())]

            self.grouped = self.data.groupby("Sentence").apply(agg_func)
            self.sentences = [s for s in self.grouped]

    getter = getsentence(data)
    sentences = getter.sentences

    sentences = sentences[0:size]

    dir = 'flair_datasets'

    files = [f'{dir}/{st.session_state[name_key][:-4]}_train.txt', f'{dir}/{st.session_state[name_key][:-4]}_test.txt',
             f'{dir}/{st.session_state[name_key][:-4]}_dev.txt']

    train, test = train_test_split(
        sentences, test_size=0.2, random_state=42, shuffle=True)

    train, dev = train_test_split(
        train, test_size=0.25, random_state=42, shuffle=True)

    datasets = [train, test, dev]

    for i in range(3):

        with open(files[i], 'w', encoding="utf-8") as f:

            for sentence in datasets[i]:

                for doc in sentence:
                    f.write(doc[0] + ' ' + doc[2])
                    f.write('\n')

                f.write('\n')

    # define columns

    columns = {0: 'text', 1: 'ner'}
    # directory where the data resides
    data_folder = 'flair_datasets'
    # initializing the corpus
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file=f'{st.session_state[name_key][:-4]}_train.txt',
                                  test_file=f'{st.session_state[name_key][:-4]}_test.txt',
                                  dev_file=f'{st.session_state[name_key][:-4]}_dev.txt')

    st.success('Dataset successfully processed')

    st.info('Training the model [it may take few hours] We recommend to use GPU device')

    # TRAINING

    tag_type = 'ner'
    # make tag dictionary from the corpus

    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

    embedding_types = [
        WordEmbeddings('glove'),
        ## other embeddings
    ]
    embeddings = StackedEmbeddings(
        embeddings=embedding_types)

    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type,
                            use_crf=True)

    trainer = ModelTrainer(tagger, corpus)

    path = f'models/flair_{st.session_state[name_key][:-4]}_{st.session_state[size_key]}'

    with st.spinner('Training the model...'):
        trainer.train(path, **paras)

    with open(f'{path}/training.log', 'r') as f:
        out = f.read()

    st.success('Training completed successfully')

    st.text_area('Console Output', out, height=300)

    st.balloons()

    st.markdown('***')

    test_df = pd.read_csv(f'{path}/test.tsv', sep=' ', engine='python',
                          header=None, usecols=[1, 2], names=['y_true', 'y_pred'],
                          encoding='utf-8', error_bad_lines=False)

    y_true = test_df.y_true.to_list()
    y_pred = test_df.y_pred.to_list()

    classes = test_df.y_true.unique()

    classes = sorted(classes)

    show_metrics(y_true=y_true, y_pred=y_pred, classes=classes)

    # # load the trained model
    # model = SequenceTagger.load(f'{path}/best-model.pt')
    # # create example sentence
    # sentence = Sentence('I love Berlin. John is in USA.')
    # # predict the tags
    # model.predict(sentence)


def stanza_train_model(data, size):

    data=data.head(size)
    word_list = data['Word'].tolist()
    ent_list = data['Tag'].tolist()
    entities = []

    nlp = spacy.blank("en")

    doc = Doc(nlp.vocab, words=word_list, ents=ent_list)

    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))

    doc_ents = [(doc.text, {'entities': entities})]

    class getsentence(object):

        def __init__(self, data):
            self.n_sent = 1.0
            self.data = data
            self.empty = False
            agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                               s["POS"].values.tolist(),
                                                               s["Tag"].values.tolist())]
            self.grouped = self.data.groupby("Sentence").apply(agg_func)
            self.sentences = [s for s in self.grouped]

    getter = getsentence(data)
    sentences = getter.sentences
    # This is how a sentence will look like.
    #print(sentences[0])



    dir = 'stanza_datasets'

    files = [f'{dir}/{st.session_state[name_key][:-4]}_train.bio', f'{dir}/{st.session_state[name_key][:-4]}_test.bio',
             f'{dir}/{st.session_state[name_key][:-4]}_dev.bio']

    train, test = train_test_split(sentences, test_size=0.2, random_state=42, shuffle=True)

    train, dev = train_test_split(train, test_size=0.25, random_state=42, shuffle=True)

    datasets = [train, test, dev]

    with open('mod_stanza_config.txt', 'r') as f:
        config = f.read()

    config_list = config.split('\n')
    PARAS = ''
    for conf in config_list:
        if '$' in conf:
            key, val = conf.split('$')
            PARAS += key + val
            PARAS += ' '

    for i in range(3):

        with open(files[i], 'w', encoding="utf-8") as f:

            for sentence in datasets[i]:

                for doc in sentence:
                    f.write(doc[0] + '\t' + doc[2])
                    f.write('\n')

                f.write('\n')

    with st.spinner('Preparing Datasets ...'):
        prepare_ner_file.process_dataset(f'{dir}/{st.session_state[name_key][:-4]}_train.bio',
                                         f'./data/ner/en_{st.session_state[name_key][:-4]}.train.json')
        prepare_ner_file.process_dataset(f'{dir}/{st.session_state[name_key][:-4]}_test.bio',
                                         f'./data/ner/en_{st.session_state[name_key][:-4]}.test.json')
        prepare_ner_file.process_dataset(f'{dir}/{st.session_state[name_key][:-4]}_dev.bio',
                                         f'./data/ner/en_{st.session_state[name_key][:-4]}.dev.json')

    cmd = f'python -m stanza.utils.training.run_ner en_{st.session_state[name_key][:-4]} {PARAS}'



    st.success('Dataset successfully processed')

    st.info('Training the model [it may take few hours] We recommend to use GPU device')

    with st.spinner('Training the model...'):
        out = check_output(cmd, shell=True)

    out = out.decode("utf-8")

    out = "\n".join(out.splitlines())

    st.success('Training completed successfully')

    st.text_area('Console Output', out, height=300)

    st.balloons()

    _,report,plot=get_report_stanza(doc_ents)

    show_train_results(report,plot)


def spacy_train_model(data, size):


    data = data.head(size)
    word_list = data['Word'].tolist()
    ent_list = data['Tag'].tolist()
    entities = []

    nlp = spacy.blank("en")

    doc = Doc(nlp.vocab, words=word_list, ents=ent_list)

    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))

    doc_ents = [(doc.text, {'entities': entities})]
    # LOAD en_core_web for finetuning and to create model from scratch use spacy.blank("en")

    ner = spacy.load(r'./en_core_web_sm-3.4.0')
    # ner=spacy.blank('en')

    dataset = []

    if st.session_state[file_type_key] == 'txt':
        dataset = process_text(data, size)
    else:
        dataset = process_df(data, size)

    test_size = int(len(dataset) * 0.30)

    test_data = dataset[0:test_size]
    train_data = dataset[test_size:]

    db_train = DocBin()
    db_test = DocBin()

    st.info('Preparing dataset for training')
    my_bar = st.progress(0)

    i = 0
    total_len = len(train_data) + len(test_data)
    for text in tqdm(train_data):
        my_bar.progress(i / total_len)
        doc = ner(text)
        db_train.add(doc)
        i += 1

    for text in tqdm(test_data):
        my_bar.progress(i / total_len)
        doc = ner(text)
        db_test.add(doc)
        i += 1

    my_bar.progress(1.0)

    db_train.to_disk(r'./datasets/train.spacy')

    db_test.to_disk(r'./datasets/dev.spacy')

    cmd = f'python -m spacy train config.cfg --output ./models/spacy_{st.session_state[name_key][:-4]}_{st.session_state[size_key]} --paths.train ./datasets/train.spacy --paths.dev ./datasets/dev.spacy'

    st.success('Dataset successfully processed')

    st.info('Training the model [it may take up to 5 minutes]')

    with st.spinner('Training the model...'):
        out = check_output(cmd, shell=True)

    out = out.decode("utf-8")

    out = "\n".join(out.splitlines())

    st.success('Training completed successfully')

    st.text_area('Console Output', out, height=300)

    st.balloons()

    _, report, plot = get_report_spacy(doc_ents)

    show_train_results(report, plot)




def model_size():

    if data_key not in st.session_state:
        st.markdown('<div class= "big center"> <b>No Dataset Selected</b> </div>', unsafe_allow_html=True)
        return

    istxt = False

    if st.session_state[file_type_key] != 'txt':
        dataset_size = len(st.session_state[data_key])
    else:
        istxt = True
        dataset_size = len(st.session_state[data_key].split('\n'))

    st.success(f'file name:- {st.session_state[name_key]}')

    if not istxt:
        st.info(f'total rows:-  {dataset_size}')
    else:
        st.info(f'total sentences:-  {dataset_size}')

    lib_select = st.selectbox(label='Select Framework',
                              options=('spacy', 'stanza', 'flair'), index=0,
                              help='Select the framework by which model should be trained')

    st.session_state[lib_key] = lib_select

    space(1)

    size_select = st.selectbox(label='Select Model Size',
                               options=('10%', '70%', '80%', '90%', '100%'), index=0,
                               help='Select the size on which model should be trained')

    st.session_state[size_key] = size_select

    col1, col2 = st.columns(2)

    subset_size = int(dataset_size * (int(size_select[0:-1]) / 100))

    with col1:

        train_size = int(subset_size * 0.7)
        # st.markdown(f'<div class="med-font"> Training Size <u>{str(train_size)}</u> </div>',unsafe_allow_html=True)
        card(title="Train Size", text=str(train_size),
             image='https://images.unsplash.com/photo-1611242320536-f12d3541249b?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxleHBsb3JlLWZlZWR8MTZ8fHxlbnwwfHx8fA%3D%3D&auto=format&fit=crop&w=600&q=60')

    with col2:
        test_size = int(subset_size * 0.3)
        # st.markdown(f'<div class="med-font"> Test Size <u>{str(test_size)}</u> </div>', unsafe_allow_html=True)
        card(title="Test Size", text=str(test_size),
             image='https://images.unsplash.com/photo-1588421357574-87938a86fa28?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxleHBsb3JlLWZlZWR8Mnx8fGVufDB8fHx8&auto=format&fit=crop&w=600&q=60')

    space(1)

    st.subheader('Hyper-Parameters')

    st.success(st.session_state[lib_key])

    if lib_select == 'spacy':

        with open('spacy_config.cfg', 'r') as f:
            spacy_config = f.read()

        spacy_content = st_ace(theme='tomorrow_night_blue', language='python', height=350, value=spacy_config)

        with open('config.cfg', 'w') as f:
            f.write(spacy_content)

    elif lib_select == 'stanza':

        st.info('Enter value with prefix as $')
        with open('stanza_config.txt', 'r') as f:
            stanza_config = f.read()

        stanza_content = st_ace(theme='tomorrow_night_blue', language='python', height=350, value=stanza_config)

        with open('mod_stanza_config.txt', 'w') as f:
            f.write(stanza_content)

    elif lib_select == 'flair':

        with open('flair_config.txt', 'r') as f:
            flair_config = f.read()

        flair_content = st_ace(theme='tomorrow_night_blue', language='python', height=350, value=flair_config)

        with open('mod_flair_config.txt', 'w') as f:
            f.write(flair_content)

    c1, c2 = st.columns([0.8, 1])

    if c2.button('RUN MODEL'):
        if lib_select == 'spacy':
            spacy_train_model(data=st.session_state[data_key], size=subset_size)

        elif lib_select == 'stanza':
            stanza_train_model(data=st.session_state[data_key], size=subset_size)

        elif lib_select == 'flair':
            flair_train_model(data=st.session_state[data_key], size=subset_size)

        else:
            st.error('Invalid lib Input')


from flair.data import Label


def pii_detection():
    models_list = os.listdir('./models')

    selected_model = st.selectbox(label='Select the model',
                                  options=(models_list), index=0,
                                  help='Select the model')

    st.info(f'{selected_model} selected')

    text = st.file_uploader('Select test file')

    # To read file as string:
    if text is not None:

        raw_text = read_txt(text, write=False)

        st.text_area('Text Preview', raw_text, disabled=False)

        st.markdown('---')

        if selected_model.startswith('stanza'):

            with st.spinner('Model Loading ...'):

                if selected_model.endswith('onto'):
                    nlp = spacy_stanza.load_pipeline('en', processors='tokenize,ner',
                                                     dir=f'models/{selected_model}',
                                                     )


                else:
                    files = os.listdir(f'models/{selected_model}')
                    pt_file = files[0]
                    nlp = spacy_stanza.load_pipeline('en', processors='tokenize,ner',
                                                     ner_model_path=f'models/{selected_model}/{pt_file}',
                                                     )

                text_ner = nlp(raw_text)

                tags_to_word = defaultdict(list)

                for ent in text_ner.ents:
                    tags_to_word[ent.label_].append(ent.text)

                col1, col2 = st.columns([2.3, 1], gap="large")

                with col1:
                    if selected_model.endswith('GMB'):
                        html = displacy.render(text_ner, style="ent", options=displacy_options)



                    else:
                        html = displacy.render(text_ner, style="ent")

                    st.markdown(html, unsafe_allow_html=True)

                with col2:
                    for tag, word_list in tags_to_word.items():
                        st.write(tag)
                        st.write(word_list)


        elif selected_model.startswith('flair'):

            with st.spinner('Model Loading ...'):

                # load the trained model

                model = None

                if selected_model.endswith('en'):
                    model = SequenceTagger.load('ner-fast')

                else:
                    model = SequenceTagger.load(f'models/{selected_model}/final-model.pt')

                # create example sentence
                sentence = Sentence(raw_text)
                # predict the tags
                model.predict(sentence)

                # print predicted NER spans

                text_tags = []
                tags_to_word = defaultdict(list)

                # iterate over entities and print

                dict_flair = sentence.to_dict(tag_type='ner')

                doc = {}
                doc['text'] = dict_flair['text']
                doc['ents'] = []

                for entity in sentence.get_spans('ner'):
                    # # print entity text, start_position and end_position
                    # print(f'entity.text is: "{entity.text}"')
                    # print(f'entity.start_position is: "{entity.start_position}"')
                    # print(f'entity.end_position is: "{entity.end_position}"')
                    #
                    # # also print the value and score of its "ner"-label
                    # print(f'entity "ner"-label value is: "{entity.get_label("ner").value}"')
                    # print(f'entity "ner"-label score is: "{entity.get_label("ner").score}"\n')

                    doc['ents'].append({
                        'start': entity.start_position,
                        'end': entity.end_position,
                        'label': entity.get_label("ner").value
                    })
                    text_tags.append((entity.text, entity.get_label("ner").value))
                    text_tags.append('   ')
                    tags_to_word[entity.get_label("ner").value].append(entity.text)

                # for entity in sentence.get_spans('ner'):
                #
                #     print(entity.text, end=' ')
                #
                #     print(entity.unlabeled_identifier)
                #
                #     text_tags.append(entity.text+' ')
                #
                #     st.write(entity)

                col1, col2 = st.columns([2.3, 1], gap="large")

                with col1:
                    html = displacy.render(doc, style="ent", manual=True)

                    st.markdown(html, unsafe_allow_html=True)

                with col2:
                    for tag, word_list in tags_to_word.items():
                        st.write(tag)
                        st.write(word_list)

























        else:
            if selected_model.endswith('core'):
                st.session_state[model_name] = f'models/{selected_model}'
                NER = spacy.load(f'models/{selected_model}')
            else:
                st.session_state[model_name] = f'models/{selected_model}/model-best'
                NER = spacy.load(f'models/{selected_model}/model-best')

            text_ner = NER(raw_text)

            tags_to_word = defaultdict(list)

            for ent in text_ner.ents:
                tags_to_word[ent.label_].append(ent.text)

            col1, col2 = st.columns([2.3, 1], gap="large")

            with col1:
                if selected_model.endswith('GMB'):
                    html = displacy.render(text_ner, style="ent", options=displacy_options)



                else:
                    html = displacy.render(text_ner, style="ent")

                st.markdown(html, unsafe_allow_html=True)

            with col2:
                for tag, word_list in tags_to_word.items():
                    st.write(tag)
                    st.write(word_list)

    # with open('data_vis.html', 'w') as f:
    #     f.write(html)
    #
    # local_html('./data_vis.html')


def show_metrics(y_true, y_pred, classes):
    report, plot = get_metrics(y_true=y_true, y_pred=y_pred, classes=classes, normalize=False)

    dataset = []

    cols = []
    cols.append('Entity')

    acc = 0.756

    for key, val in report.items():

        for k in val.keys():
            cols.append(k)

        break

    for key, val in report.items():
        row = []
        row.append(key)

        if type(val) != dict:
            acc = val
            break

        for k, v in val.items():
            row.append(v)

        dataset.append(row)

    df = pd.DataFrame(dataset, columns=cols)

    st.table(df)
    st.success(f'Accuracy:- {acc}')
    st.write('***')
    st.pyplot(plot)

def show_train_results(report,plot):
    df,acc=report
    st.table(df)
    st.success(f'Accuracy:- {acc}')
    st.write('***')
    st.pyplot(plot)





from spacy.tokens import Doc



def get_report_spacy(docs,model_name='models/spacy_GMB/model-best'):

    report, cm = evaluate.eval_spacy(docs, model_name=model_name)

    dataset = []

    cols = []
    cols.append('Entity')

    acc = 0.756

    for key, val in report.items():

        for k in val.keys():
            cols.append(k)

        break

    for key, val in report.items():
        row = []
        row.append(key)

        if type(val) != dict:
            acc = val
            break

        for k, v in val.items():
            row.append(v)

        dataset.append(row)

    report = pd.DataFrame(dataset, columns=cols)

    #st.dataframe(report)

    spacy_recall = report['recall'].iloc[-1]
    spacy_precison = report['precision'].iloc[-1]
    spacy_f1 = report['f1-score'].iloc[-1]

    spacy_metrics = ["Spacy", spacy_f1 * 100, spacy_precison * 100, spacy_recall * 100]

    st.write(spacy_metrics)

    return spacy_metrics,(report,acc),cm



def get_report_stanza(docs,model_name='models/stanza_onto'):

    report, cm = evaluate.eval_stanza(docs, model_name=model_name)

    dataset = []

    cols = []
    cols.append('Entity')

    acc = 0.756

    for key, val in report.items():

        for k in val.keys():
            cols.append(k)

        break

    for key, val in report.items():
        row = []
        row.append(key)

        if type(val) != dict:
            acc = val
            break

        for k, v in val.items():
            row.append(v)

        dataset.append(row)

    report = pd.DataFrame(dataset, columns=cols)



    #st.dataframe(report)

    recall = report['recall'].iloc[-1]
    precison = report['precision'].iloc[-1]
    f1 = report['f1-score'].iloc[-1]

    stanza_metrics = ["Stanza", f1 * 100, precison * 100, recall * 100]

    st.write(stanza_metrics)

    return stanza_metrics,(report,acc),cm


def get_report_flair():
    test_df = pd.read_csv('models/flair_GMB/test.tsv', sep=' ', engine='python',
                          header=None, usecols=[1, 2], names=['y_true', 'y_pred'],
                          encoding='utf-8', error_bad_lines=False)

    y_true = test_df.y_true.to_list()
    y_pred = test_df.y_pred.to_list()

    classes = test_df.y_true.unique()

    classes = sorted(classes)

    report, cm = get_metrics(y_true=y_true, y_pred=y_pred, classes=classes, normalize=False)

    dataset = []

    cols = []
    cols.append('Entity')

    acc = 0.756

    for key, val in report.items():

        for k in val.keys():
            cols.append(k)

        break

    for key, val in report.items():
        row = []
        row.append(key)

        if type(val) != dict:
            acc = val
            break

        for k, v in val.items():
            row.append(v)

        dataset.append(row)

    report = pd.DataFrame(dataset, columns=cols)

    #st.dataframe(report)

    recall = report['recall'].iloc[-1]
    precison = report['precision'].iloc[-1]
    f1 = report['f1-score'].iloc[-1]

    flair_metrics = [ "Flair",f1 * 100, precison * 100, recall * 100]

    st.write(flair_metrics)

    return flair_metrics,(report,acc),cm




def plotbar():
    st.title('Analysis')

    st.markdown('---')

    file = st.file_uploader('Choose a file', type=['csv', 'xlsx', 'xls', 'txt', 'tsv'],
                             accept_multiple_files=False)

    # if 'files' not in st.session_state:
    #     st.session_state.files = files

    if file is not None:

        st.write('Preview')

        file_type = file.name.partition('.')[-1]

        st.caption(f'.{file_type} file')

        if file_type == 'csv' or file_type == 'xls' or file_type == 'xlsx':
            data = read_csv(file)
        elif file_type == 'tsv':
            data = read_tsv(file)
        else:
            data = read_txt(file)



        data=data.tail(1000)





        word_list=data['Word'].tolist()
        ent_list=data['Tag'].tolist()
        entities = []

        nlp = spacy.blank("en")

        doc = Doc(nlp.vocab, words=word_list, ents=ent_list)



        for ent in doc.ents:
            entities.append((ent.start_char, ent.end_char, ent.label_))





        doc_ents = [(doc.text, {'entities': entities})]


        with st.spinner('getting spacy report'):
            spacy_metrics, _, __ = get_report_spacy(doc_ents)



        with st.spinner('getting stanza report'):
            stanza_metrics, _, __ = get_report_stanza(doc_ents)

        with st.spinner('getting flair report'):
           flair_metrics, _, __ = get_report_flair()



        # continue loading the data with your excel file, I was a bit too lazy to build an Excel file :)
        df = pd.DataFrame(
            [
               spacy_metrics,
                stanza_metrics,
                flair_metrics


            ],

            columns=["Framework",  "F1", "Precision", "Recall"]
        )

        fig = px.bar(df, x="Framework", y=["F1", "Precision", "Recall"], barmode='group', height=500,color_discrete_sequence=['red','blue','yellow'])
        # st.dataframe(df) # if need to display dataframe
        st.plotly_chart(fig)




def results():

    models_list = os.listdir('./models')

    selected_model = st.selectbox(label='Select the model',
                                  options=(models_list), index=0,
                                  help='Select the model')

    st.success(f'{selected_model} selected')

    text = st.file_uploader('Select test file')

    # To read file as string:
    if text is not None:

        raw_text = read_txt(text, write=False)

        st.text_area('Text Preview', raw_text, disabled=False)

        if selected_model.startswith('stanza'):
            st.title('Work in progress for Stanza Evaluation')
            return


        else:
            if selected_model.endswith('core'):
                st.session_state[model_name] = f'models/{selected_model}'

            else:
                st.session_state[model_name] = f'models/{selected_model}/model-best'

        st.markdown('***')
        docs = getdoc(raw_text)

        report, plot = evaluate.eval_spacy(docs, model_name=st.session_state[model_name])

        dataset = []

        cols = []
        cols.append('Entity')

        acc = 0.756

        for key, val in report.items():

            for k in val.keys():
                cols.append(k)

            break

        for key, val in report.items():
            row = []
            row.append(key)

            if type(val) != dict:
                acc = val
                break

            for k, v in val.items():
                row.append(v)

            dataset.append(row)

        df = pd.DataFrame(dataset, columns=cols)

        st.table(df)
        st.success(f'Accuracy:- {acc}')
        st.write('***')
        st.pyplot(plot)


from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from faker import Faker
from fpdf import FPDF

from presidio_anonymizer.entities.engine import OperatorConfig
from textwrap3 import wrap as textwrap

def text_to_pdf(text, filename):
    a4_width_mm = 210
    pt_to_mm = 0.35
    fontsize_pt = 10
    fontsize_mm = fontsize_pt * pt_to_mm
    margin_bottom_mm = 10
    character_width_mm = 7 * pt_to_mm
    width_text = a4_width_mm / character_width_mm

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(True, margin=margin_bottom_mm)
    pdf.add_page()
    pdf.set_font(family='Courier', size=fontsize_pt)
    splitted = text.split('\n')

    for line in splitted:
        lines = textwrap(line, width_text)

        if len(lines) == 0:
            pdf.ln()

        for wrap in lines:
            pdf.cell(0, fontsize_mm, wrap, ln=1)

    pdf.output(filename, 'F')

def mask_PII():
    text_area_height=150

    fake=Faker()

    all_ents = [
        "CREDIT_CARD",
        "CRYPTO",
        "DATE_TIME",
        "EMAIL_ADDRESS",
        "IBAN_CODE",
        "IP_ADDRESS",
        "NRP",
        "LOCATION",
        "PERSON",
        "PHONE_NUMBER",
        "MEDICAL_LICENSE",
        "URL",

    ]

    fake_ents={
        all_ents[0]:fake.credit_card_number,   #CREDIT_CARD
        all_ents[1]:None,                        #CRYPTO
        all_ents[2]: fake.date,                 #DATE_TIME
        all_ents[3]:fake.email,                   #EMAIL_ADDRESS
        all_ents[4]:fake.ssn,                    # IBAN_CODE
        all_ents[5]: None,                      #IP_ADDRESS
        all_ents[6]: fake.country,                      # NRP
        all_ents[7]: fake.country,                      #LOCATION,
        all_ents[8]:fake.name,                      #PERSON,
        all_ents[9]: fake.phone_number,                      #PHONE_NUMBER,
        all_ents[10]: None,                      #MEDICAL_LICENSE
        all_ents[11]: None, #url



    }

    masking_levels = ['low', 'medium', 'high']

    levels_count = {
        masking_levels[0]: 2,
        masking_levels[1]: 4,
        masking_levels[2]: 8,
    }

    masking_options = ['Category Mask', 'Normal Masking', 'Pseudo-Masking']






    text_input = st.text_area("Type a text to anonymize")

    uploaded_file = st.file_uploader("or Upload a file", type=["doc", "docx", "pdf", "txt"])

    if uploaded_file is not None:
        text_input = uploaded_file.getvalue()
        text_input = text_input.decode("utf-8")


    st.text_area('Text preview',text_input,height=text_area_height)






    selected_ents=st.multiselect('Which PII do you want to mask', options=all_ents,default=all_ents)



    masking_scheme=st.selectbox('Select masking scheme', options=masking_options,index=2)



    if masking_options[1] == masking_scheme:
        masking_level = st.selectbox('Level of masking', options=masking_levels,index=0)

    operator_options={}

    no_mask=OperatorConfig(operator_name="mask", params={'chars_to_mask': 0,
                                                        'masking_char': '*',
                                                        'from_end': True})

    category_replace=OperatorConfig(operator_name="replace")






    if masking_scheme==masking_options[0]:

        st.info('Replace by Category')

        for ent in all_ents:

            if ent in selected_ents:
                operator_options[ent]=category_replace
            else:
                operator_options[ent]=no_mask



    elif masking_scheme==masking_options[1]:

        st.info('Mask by *')

        for ent in all_ents:

            if ent in selected_ents:
                operator_options[ent] =OperatorConfig(operator_name="mask",
                                            params={'chars_to_mask': levels_count[masking_level],
                                                    'masking_char': '*',
                                                    'from_end': True})
            else:
                operator_options[ent] = no_mask

    else:

        st.info('Pseudo Masking')


        operator_options={

           all_ents[0]: OperatorConfig("custom", {"lambda": lambda x: fake.credit_card_number()}),
            all_ents[1]: OperatorConfig("custom", {"lambda": lambda x: fake.fake.credit_card_number()}),
            all_ents[2]: OperatorConfig("custom", {"lambda": lambda x: fake.date()}),
            all_ents[3]: OperatorConfig("custom", {"lambda": lambda x: fake.email()}),
            all_ents[4]: OperatorConfig("custom", {"lambda": lambda x: fake.ssn()}),
            all_ents[5]: OperatorConfig("replace", params={'new_value':'165.10.7.1'}),
            all_ents[6]: OperatorConfig("custom", {"lambda": lambda x: fake.country()}),
            all_ents[7]: OperatorConfig("custom", {"lambda": lambda x: fake.country()}),
            all_ents[8]: OperatorConfig("custom", {"lambda": lambda x: fake.name()}),
            all_ents[9]: OperatorConfig("custom", {"lambda": lambda x: fake.phone_number()}),
            all_ents[10]: OperatorConfig("custom", {"lambda": lambda x: fake.ssn()}),
            all_ents[11]: OperatorConfig("custom", {"lambda": lambda x: fake.url()}),





        }


















    # Create configuration containing engine name and models
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }

    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)

    nlp_engine = provider.create_engine()

    # the languages are needed to load country-specific recognizers
    # for finding phones, passport numbers, etc.
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine,
                              supported_languages=["en"])

    # language is a required parameter. So if you don't know
    # the language of each particular text, use language detector
    results = analyzer.analyze(text=text_input,
                               language='en')




    anonymizer = AnonymizerEngine()






    anonymized_text = anonymizer.anonymize(text=text_input,
                                           analyzer_results=results,
                                           operators=operator_options).text













    st.text_area('Output',anonymized_text,height=text_area_height)

    with open("Mask_Output.txt", "w") as text_file:
        text_file.write(anonymized_text)



    # save FPDF() class into
    # a variable pdf
    input_filename = 'Mask_Output.txt'
    output_filename = 'Mask_Output.pdf'
    file = open(input_filename)
    text = file.read()
    file.close()
    text_to_pdf(text, output_filename)





    '''


    CREDIT_CARD	A credit card number is between 12 to 19 digits. https://en.wikipedia.org/wiki/Payment_card_number	Pattern match and checksum
    CRYPTO	A Crypto wallet number. Currently only Bitcoin address is supported	Pattern match, context and checksum
    DATE_TIME	Absolute or relative dates or periods or times smaller than a day.	Pattern match and context
    EMAIL_ADDRESS	An email address identifies an email box to which email messages are delivered	Pattern match, context and RFC-822 validation
    IBAN_CODE	The International Bank Account Number (IBAN) is an internationally agreed system of identifying bank accounts across national borders to facilitate the communication and processing of cross border transactions with a reduced risk of transcription errors.	Pattern match, context and checksum
    IP_ADDRESS	An Internet Protocol (IP) address (either IPv4 or IPv6).	Pattern match, context and checksum
    NRP	A person’s Nationality, religious or political group.	Custom logic and context
    LOCATION	Name of politically or geographically defined location (cities, provinces, countries, international regions, bodies of water, mountains	Custom logic and context
    PERSON	A full person name, which can include first names, middle names or initials, and last names.	Custom logic and context
    PHONE_NUMBER	A telephone number	Custom logic, pattern match and context
    MEDICAL_LICENSE	Common medical license numbers.	Pattern match, context and checksum
    URL	A URL (Uniform Resource Locator), unique identifier used to locate a resource on the Internet	Pattern match, context and top level url validation




    operators = {"PERSON": OperatorConfig(operator_name="replace",
                                          params={"new_value": "REPLACED_NAME"}),

                 "LOCATION": OperatorConfig(operator_name="mask",
                                            params={'chars_to_mask': 3,
                                                    'masking_char': '*',
                                                    'from_end': True}),

                 # Partial Masking by giving masking char count

                 "DEFAULT": OperatorConfig(operator_name="redact")}
                 
                 
                 
                 
      fake_operators = {
        "PERSON": OperatorConfig("custom", {"lambda": lambda x: fake.name()}),
        "PHONE_NUMBER": OperatorConfig("custom", {"lambda": lambda x: fake.phone_number()}),
        "EMAIL_ADDRESS": OperatorConfig("custom", {"lambda": lambda x: fake.email()}),
        "LOCATION": OperatorConfig("custom", {"lambda": lambda x: fake.country()}),

        # OperatorConfig("replace", {"new_value": "USA"}),
        "DEFAULT": OperatorConfig(operator_name="mask",
                                  params={'chars_to_mask': 3,
                                          'masking_char': '*',
                                          'from_end': False}),
    }
                 
                 
                https://zetcode.com/python/faker/ for all categories
                 
     
      print('Mask by category')
     
     
     
                 
    '''








