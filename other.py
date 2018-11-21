# -*- coding: utf-8 -*-

import glob
import re
import json
import requests
import pandas as pd

path = './data/*.txt'
data_path = './data/mqa.csv'

shards = glob.glob(path)

data = []

def extract_question(line_list):
    ful_q = ' '.join(line_list[1:])
    question = re.sub('没有答案|（没有答案）|答案错误', '', ful_q)
    return question.strip()
    

for shard in shards:
    with open(shard, encoding='utf-8-sig') as fout:
        for idx, line in enumerate(fout):

            line_list = line.split()

            if '原始问题' in line_list:
                origin_question = extract_question(line_list)
            if '扩展问题' in line_list:
                kz_question = extract_question(line_list)
                sample = {'q1': origin_question, 
                          'q2': kz_question, 
                          'match': 1}
                data.append(sample)


origin_question_set = set()
all_question_set = set()
for sample in data:
    origin_question = sample['q1']
    kz_question = sample['q2']
    # origin_questions.append(origin_question)
    origin_question_set.add(origin_question)
    all_question_set.add(origin_question)
    all_question_set.add(kz_question)
    
origin_questions = list(origin_question_set)


url = 'http://192.168.1.145:8699/qa/ch6/?question={}'

response_data = json.loads(requests.get(url.format('黑木耳饮食宜忌百科')).text)['data']['question_parse']


response_data_questions = json.loads(requests.get(url.format('宜忌')).text)['data']['result']['ch6']['message']


for question in origin_questions:
    idf_key_words = json.loads(requests.get(url.format(question)).text)['data']['question_parse']['idf_key_words']
    
    idf_idx = 0
    response_data_questions = []
    while not len(response_data_questions) and idf_idx < len(idf_key_words):
        main_word = idf_key_words[idf_idx]
        response_data_questions = json.loads(requests.get(url.format(main_word)).text)['data']['result']['ch6']['message']
        idf_idx += 1
    
    for r_q in response_data_questions[:10]:
        if r_q not in all_question_set:
            sample = {'q1': question, 
                      'q2': r_q, 
                      'match': 0}
            data.append(sample)
            
df = pd.DataFrame(data)

df.to_csv(data_path)


# json.dump(data, open(data_path, 'w', encoding='utf-8'))
