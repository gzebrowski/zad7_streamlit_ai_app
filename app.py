import json
import math
import random
import re
from pathlib import Path
from typing import Optional

import instructor
import numpy as np
import pandas as pd
import streamlit as st
from audiorecorder import audiorecorder
from faker import Faker
from openai import OpenAI
from pycaret.clustering import (assign_model, create_model, load_model,
                                predict_model, save_model)
from pycaret.clustering import setup as setup_model
# from pydantic.main import ModelMetaclass
from pydantic import BaseModel

from helpers import (get_mp3_audio_and_hash, to_int, transcribe_audio,
                     write_html)
from qdrant_helpers import QdrantHelper
from streamlit_env import Env
from dynamic_filters import DynamicFilters

env = Env('.env')
OPENAI_API_KEY = env["OPENAI_API_KEY"]

EDU_LEVEL_CHOICES = ['Podstawowe', 'Średnie', 'Wyższe'], [0.13, 0.53, 0.34]
FAV_PLACES_CHOICES = ['nad wodą', 'w lesie', 'w górach', 'inne'], [0.3, 0.15, 0.45, 0.1]
FAV_ACTIVITIES = [
    ('filmy', 'movies'),
    ('muzyka', 'music'),
    ('książki', 'books'),
    ('spacery', 'walking'),
    ('jazda na rowerze', 'cycling'),
    ('kino', 'cinema'),
    ('teatr', 'theathre'),
    ('taniec', 'dance'),
    ('gry komputerowe', 'computer_games'),
    ('sport', 'sport'),
    ('fintess', 'fitness'),
    ('malowanie', 'painting'),
    ('wolontariat', 'voluntary_service'),
    ('fotografia', 'photography'),
    ('inne', 'other'),
]

FAITH_CHOICES = [
    ['katolik praktykujący', 'katolik niepraktykujący', 'protestant / ewangeliczny chrześcijanin',
        'ateista', 'prawosławny', 'inne wyznania'],
    [0.30, 0.50, 0.01, 0.12, 0.01, 0.06]
]
POLITICAL_CONVICTIONS_FAITH_RELATION = [
    ('lewicowy', (0.01, 0.2, 0.01, 0.4, 0.01, 0.01)),
    ('liberalny', (0.24, 0.6, 0.24, 0.59, 0.24, 0.24)),
    ('prawicowy', (0.75, 0.2, 0.75, 0.01, 0.75, 0.75)),
]


def get_age_category(age: Optional[int]) -> str:
    if not age:
        return None
    age_category_tmp = (age - 18) // 5
    return '%s-%s' % (age_category_tmp * 5 + 18, (age_category_tmp + 1) * 5 + 18)


def get_or_create_fake_corelated_data(filename):
    try:
        people = pd.read_csv(filename, sep=';')
    except FileNotFoundError:
        pass
    else:
        return people
    faker = Faker('pl')
    people = []
    favorite_correlations = {
        'movies': {'w': 0.3},
        'music': {'w': 0.15},
        'books': {'w': 0.5, 'edu_level': (0.1, 0.3, 0.6), 'sex': (0.7, 0.3)},
        'walking': {'w': 0.3},
        'cycling': {'w': 1, 'bmi': (0.6, 0.3, 0.1)},
        'cinema': {'w': 0.20},
        'theathre': {'w': 0.3, 'edu_level': (0.1, 0.3, 0.6), 'sex': (0.6, 0.4)},
        'dance': {'w': 2.5, 'edu_level': (0.45, 0.3, 0.25), 'bmi': (0.5, 0.3, 0.2), 'sex': (0.6, 0.4)},
        'computer_games': {'w': 0.8, 'edu_level': (0.6, 0.3, 0.1), 'bmi': (0.2, 0.3, 0.5), 'sex': (0.1, 0.9)},
        'sport': {'w': 0.5, 'bmi': (0.6, 0.3, 0.1)},
        'fitness': {'w': 0.4, 'bmi': (0.6, 0.3, 0.1), 'sex': (0.9, 0.1)},
        'painting': {'w': 0.15, 'sex': (0.7, 0.3)},
        'voluntary_service': {'w': 0.05},
        'photography': {'w': 0.15},
        'other': {'w': 0.2}
    }
    fav_places_n, fav_places_v = FAV_PLACES_CHOICES
    rows_count = int(env.get('ROWS_NUMBER', 2000))
    for _ in range(rows_count):
        sex = 'mężczyzna' if np.random.random(1)[0] > 0.55 else 'kobieta'
        height_params = [165, 6, 1] if sex == 'kobieta' else [180, 7, 1]
        height = int(np.random.normal(*height_params)[0])
        bmi = np.random.normal(26.5, 4.5, 1)[0]
        bmi_level = max(0, min(2, (bmi // 5) - 4))  # should be 0, 1 or 2
        weight = int(bmi * math.pow((height / 100), 2))
        first_name = str(faker.first_name_female() if sex == 'kobieta' else faker.first_name_male())
        edu_level_nr = np.random.choice([0, 1, 2], p=EDU_LEVEL_CHOICES[1])
        edu_level = EDU_LEVEL_CHOICES[0][edu_level_nr]
        sex_nr = 0 if sex == 'kobieta' else 1
        faith_data = FAITH_CHOICES
        political_convictions_faith_relation = POLITICAL_CONVICTIONS_FAITH_RELATION
        religious_faith = np.random.choice(len(faith_data[0]), 1, p=faith_data[1])[0]
        political_convictions_p_param = [x[1][religious_faith] for x in political_convictions_faith_relation]
        political_convictions = np.random.choice(
            len(political_convictions_faith_relation), 1,
            p=political_convictions_p_param)[0]
        item = {
            'age': 18 + int(max(0, np.random.normal(18, 8, 1)[0])),
            'first_name': first_name,
            'last_name': faker.last_name(),
            'sex': sex,
            'height': height,
            'weight': weight,
            'edu_level': edu_level,
            'religious faith': faith_data[0][religious_faith],
            'political choice': political_convictions_faith_relation[political_convictions][0],
            'fav_place': np.random.choice(fav_places_n, 1, p=fav_places_v)[0],
        }
        item['age_category'] = get_age_category(int(item['age']))
        factor_levels = {'bmi': bmi_level, edu_level: edu_level_nr, 'sex': sex_nr}
        fl_keys = list(factor_levels.keys())
        for fav, fav_factors in favorite_correlations.items():
            p = float(fav_factors['w'])
            for fl_key in fl_keys:
                if fav_factor := fav_factors.get(fl_key):
                    p *= fav_factor[int(factor_levels[fl_key])]
            item[f'fav_{fav}'] = 1 if random.random() <= p else 0
        people.append(item)
    people = pd.DataFrame(people, index=list(range(1, rows_count + 1)))
    people.to_csv(filename, sep=';', index=None)
    return people


@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)


def generate_cluster_descriptions(data_with_kmeans):
    path = Path(env['CLUSTER_DESCRIPTIONS_PATH'])
    if path.exists():
        return json.loads(open(path).read())
    all_clusters = data_with_kmeans['Cluster'].unique()

    data = {}
    for clstr in all_clusters:
        df: pd.DataFrame = data_with_kmeans[data_with_kmeans['Cluster'] == clstr]
        c_k = str(clstr)
        data[c_k] = []
        for column in df.columns:
            if column == 'Cluster':
                continue
            value_counts = df.value_counts(subset=column)
            data[c_k].append([column, ', '.join([f'{c}: {cnt}' for c, cnt in value_counts.items()])])
    summary = '\n\n'.join([(f'Klaster "{claster}":\n' + '\n'.join([f'{c}: {d}' for c, d in lines]))
                           for claster, lines in data.items()])
    prompt = f'Użyliśmy algorytmu klastrowania\n{summary}\n\n'\
             'Wygeneruj najlepsze nazwy dla każdego z klastrów oraz ich opisy\n\nUżyj formatu JSON. Przykładowo:\n'\
             '''
                {
                    "Cluster 0": {
                        "name": "<krótka nazwa Klastra 0>",
                        "description": "W tej kategorii znajdują się osoby, które..."
                    }
                    "Cluster 1": {
                        "name": "<krótka nazwa Klastra 1>",
                        "description": "W tej kategorii znajdują się osoby, które..."
                    },
                    ...
                }
    '''
    openai_client = get_openai_client()

    response = openai_client.chat.completions.create(model=env['DESC_AI_MODEL'], temperature=0, messages=[{
        'role': 'user',
        'content': [{'type': 'text', 'text': prompt}],
    }])
    result = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
    cluster_names_desc = json.loads(result)
    with open(path, 'w') as f:
        f.write(json.dumps(cluster_names_desc))
    return cluster_names_desc


def load_my_model(try_again=False):
    my_model_name = env['MY_MODEL_NAME']
    path = Path(f'{my_model_name}.pkl')
    people_data_df = get_or_create_fake_corelated_data(Path(env['PEOPLE_DATA_FILENAME']))
    columns = ['sex', 'edu_level', 'religious faith', 'political choice', 'age_category']
    columns += [c for c in people_data_df.columns if c.startswith('fav_')]
    df = people_data_df[columns]
    setup_model(df, session_id=env.get('MODEL_SESSION_ID', 123))
    if path.exists():
        kmeans_pipeline = load_model(my_model_name)
        descs = generate_cluster_descriptions(kmeans_pipeline)
        data_with_kmeans = assign_model(kmeans_pipeline)
        return kmeans_pipeline, descs, data_with_kmeans, people_data_df
    if try_again:
        raise RuntimeError
    kmeans = create_model('kmeans', num_clusters=16)
    data_with_kmeans = assign_model(kmeans)
    generate_cluster_descriptions(data_with_kmeans)
    save_model(kmeans, my_model_name, verbose=False)
    return load_my_model(True)


def find_cluster(kmeans_pipeline, data):
    predict_with_cluster_df = predict_model(model=kmeans_pipeline, data=data)
    return predict_with_cluster_df['Cluster']


def retrieve_structure(openai_client, text, response_model):
    instructor_openai_client = instructor.from_openai(openai_client)
    res = instructor_openai_client.chat.completions.create(
        model=env['DESC_AI_MODEL'],
        temperature=0,
        response_model=response_model,
        messages=[
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    return res.model_dump()


qdrant_helper = QdrantHelper(get_openai_client())

kmeans_pipeline, cluster_names_and_descriptions, search_data, full_data = load_my_model()
dyn_f = DynamicFilters(full_data)

if 'gathered_data' not in st.session_state:
    st.session_state.gathered_data = {}
    st.session_state.audio_hashes = []
    st.session_state.audio_story_hash = None

data_to_gather: list[tuple[str, str, dict]] = [
    ('sex', 'podaj swoją płeć', {'prefix': 'Płeć ', 'values': ['mężczyzna', 'kobieta']}),
    ('edu_level', 'podaj wykształcenie', {'prefix': 'wykształcenie ', 'values': EDU_LEVEL_CHOICES[0]}),
    ('fav_place', 'podaj, gdzie najbardziej lubisz wypoczywać', {'prefix': 'Ulubione miejsce wypoczynku ',
                                                                 'values': FAV_PLACES_CHOICES[0]}),
    ('fav_activity', 'podaj, ulubiony sposób spędzania wolnego czasu',
     {'prefix': 'Ulubiony sposób spędzania wolnego czasu ', 'values': [f[0] for f in FAV_ACTIVITIES]}),
    ('religious faith', 'podaj wyznanie', {'prefix': 'Wyznanie ', 'values': FAITH_CHOICES[0]}),
    ('political choice', 'podaj przekonania polityczne',
     {'prefix': 'opcja polityczna ', 'values': [x[0] for x in POLITICAL_CONVICTIONS_FAITH_RELATION]}),
    ('age_category', 'podaj wiek', {'numeric': True, 'format': get_age_category, 'help_text': 'Podaj samą liczbę lat'}),
]
qdrant_helper.index_embedings([
    {
        'name': x[0],
        'values': [x[2]['prefix'] + v for v in x[2]['values']],
    } for x in data_to_gather if x[2].get('values')])

with st.sidebar:
    dyn_f.show_filters()

tab1, tab2, tab3 = st.tabs(['znajdź dopasowania', 'eksploruj dane', 'Opowiedz o sobie'])
with tab1:
    gathered_data = dict(st.session_state.gathered_data)
    any_missing = False
    for k, prompt, opts in data_to_gather:
        if k not in gathered_data:
            any_missing = True
            st.write(prompt)
            help_txt = ''
            if opts.get('values'):
                help_txt = 'np: %s' % ', '.join(opts['values'])
            elif opts.get('help_text'):
                help_txt = opts['help_text']
            write_html(f'<p style="color:#888; font-size:0.7em;"><em>{help_txt}</em></p>')
            curr_rec = audiorecorder(start_prompt='Nagraj', stop_prompt='Zakończ')
            if curr_rec:
                audio_bytes, hsh = get_mp3_audio_and_hash(curr_rec)
                if hsh not in st.session_state.audio_hashes:
                    st.session_state.audio_hashes = st.session_state.audio_hashes + [hsh]
                    txt = transcribe_audio(get_openai_client(), audio_bytes)
                    if opts.get('values'):
                        recs = qdrant_helper.search_values_from_db(k, query=txt, limit=2)
                        gathered_data[k] = txt, recs[0]['text'][len(opts['prefix']):], recs
                    elif opts.get('numeric'):
                        val = re.search(r'[\d]+', txt)
                        val = to_int(val[0] if val else None)
                        val = opts['format'](val)
                        gathered_data[k] = txt, val, None
                    st.session_state.gathered_data = gathered_data
                    st.rerun()
                    # st.audio(audio_bytes, format="audio/mp3")
            break
        else:
            st.write(f'**{prompt}**: ', str(st.session_state.gathered_data[k]))

    if not any_missing:
        fav_activity = gathered_data.pop('fav_activity')
        data_person = {
            k: gathered_data[k][1] for k in gathered_data
        }
        for f_pol, f_field in FAV_ACTIVITIES:
            data_person[f'fav_{f_field}'] = 1 if f_pol == fav_activity[1] else 0

        person_df = pd.DataFrame([data_person])
        predicted_cluster_id = predict_model(kmeans_pipeline, data=person_df)["Cluster"].values[0]
        predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]
        st.header('Rezultat')
        write_html(f"<p><strong>Najbardziej dopasowana kategoria:</strong> {predicted_cluster_data['name']}</p>")
        write_html(f"<p>{predicted_cluster_data['description']}</p>")
        idxs = search_data[search_data['Cluster'] == predicted_cluster_id].index.to_list()
        st.header('Lista osób z tej kategorii')
        st.write(full_data[full_data.index.isin(idxs)])


with tab2:
    dyn_f.show_dataframe()


fields = [
    ('gender', 'sex', str),
    ('favorite_place', 'fav_place', str),
    ('education_level', 'edu_level', str),
    ('favorite_activity', 'fav_activity', str),
    ('religious_faith', 'religious faith', str),
    ('political_choice', 'politicalchoice', str),
    ('age', 'age', int),
]

with tab3:
    cls_fields = {}
    cls_annotations = {}

    for f, gf, tp in fields:
        if gf not in gathered_data:
            cls_fields[f] = None
            cls_annotations[f] = Optional[tp]

    cls_fields['__annotations__'] = cls_annotations
    MyModel = type('MyModel', (BaseModel,), cls_fields)
    my_model = MyModel()
    st.header('Opowiedz nam o sobie')
    st.write(
        '''podaj swoją płeć, swój wiek, swoje wykształcenie, ulubione miejsce wypoczynku,
ulubiony sposób spędzania wolnego czasu, swoje przekonania religijne i polityczne''')
    curr_rec_story = audiorecorder(start_prompt='Nagraj', stop_prompt='Zakończ', key='audio_story')
    if curr_rec_story:
        audio_bytes_story, hsh2 = get_mp3_audio_and_hash(curr_rec_story)
        if hsh2 != st.session_state.audio_story_hash:
            st.session_state.audio_story_hash = hsh2
            txt2 = transcribe_audio(get_openai_client(), audio_bytes_story)
            result3 = retrieve_structure(get_openai_client(), txt2, MyModel)
            st.write(result3)
