from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
import string
from sklearn.cluster import KMeans
import time
import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
import random
from itertools import combinations


tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')


def make_summarization(sentences, tokenizer, model): # 문자열 생성 요약
    if(len(sentences) < 4): return max(sentences, key=lambda x:len(x))
    split = []
    for i in range(len(sentences)//8):
        split.append(sentences[:8])
        sentences = sentences[8:]

    for i in range(len(split)):
        if(len(sentences) == 0): break
        split[i].append(sentences.pop())
    
    if(len(sentences) != 0): split.append(sentences)
    
    split_sum = []
    for sentences in split:
        text = '\n'.join(sentences)
        start = time.time()
        raw_input_ids = tokenizer.encode(text)
        input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

        summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=256, min_length=48,  eos_token_id=1)
        print(f"{time.time()-start:.4f} sec")
        sum_result = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        print(sum_result)
        split_sum.append(sum_result)
        print(len(split), len(split_sum))
        print('-----------------------------------------------')
    
    if(len(split_sum)==1):
        return split_sum[0]
      
    return ' '.join(split_sum)


def summarize_topic(document_df, topic_num, tokenizer, model): # df와 topic num을 넣으면 해당 topic num을 요약
    sentences = []
    numbers = []
    for i,t in enumerate(document_df[document_df['cluster_label'] == topic_num]['opinion_text']):
        sentences.append(eval(t)[1])
        numbers.append(eval(t)[0])
    result = make_summarization(sentences, tokenizer, model)
    avg = sum(numbers)/len(numbers)
    return (avg, result)


def summarize_first_sentences(processed_sentences, tokenizer, model): # 문쟈열을 k-means로 토픽 별 분류(첫 문장)
    document_df = get_clustered_df(processed_sentences, clusternum)
    sum_result = []
    temp = get_topic_sentences(document_df, 1)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    summ_result = summarize_topic(document_df, 1, tokenizer, model)
    print(summ_result)
    print('===================================================')
    
    first_result = (sum_result[0], max(sum_result[1].split('. '), key=lambda x:len(x)))
    return first_result
    

def summarize_topk_sentences(processed_sentences, tokenizer, model): # 문쟈열을 k-means로 토픽 별 분류
    clusternum = len(processed_sentences)//7
    document_df = get_clustered_df(processed_sentences, clusternum)
    sum_result = []
    for c in range(clusternum):
        temp = get_topic_sentences(document_df, c)
        print('---------------------------------------------------')
        summ = summarize_topic(document_df, c, tokenizer, model)
        sum_result.append(summ)
        print(summ)
        print('***************************************************')
        
    return sorted(sum_result, key= lambda x: x[0])


def get_clustered_df(sentences, clusternum):
    print(clusternum)
    document_df = pd.DataFrame()
    document_df['opinion_text'] = [str(t) for t in sentences]
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

    lemmar = WordNetLemmatizer()
    
    # 토큰화한 각 단어들의 원형들을 리스트로 담아서 반환
    def LemTokens(tokens):
        return [lemmar.lemmatize(token) for token in tokens]

    # 텍스트를 Input으로 넣어서 토큰화시키고 토큰화된 단어들의 원형들을 리스트로 담아 반환
    def LemNormalize(text):
        return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    
    tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize,
                            ngram_range=(1,2),
                            min_df=0.05, max_df=0.85)


    ftr_vect = tfidf_vect.fit_transform(document_df['opinion_text'])
    kmeans = KMeans(n_clusters=clusternum, max_iter=10000, random_state=42)
    cluster_label = kmeans.fit_predict(ftr_vect)
    
    # 군집화한 레이블값들을 document_df 에 추가하기
    document_df['cluster_label'] = cluster_label
    return document_df.sort_values(by=['cluster_label'])
    

def get_topic_sentences(df, clabel):
    lst = []
    for i,t in enumerate(df[df['cluster_label'] == clabel]['opinion_text']):
        print(i, t)
        lst.append(t)
    return lst 


def delete_similar(input_sentences):
    sorted_sentences = sorted(input_sentences, key=lambda x:x[1][::-1])
    prev_len = len(sorted_sentences)
    
    for i in range(5):
        prev = sorted_sentences[0]
        processed_sentences = []
        for j,sentence in enumerate(sorted_sentences[1:]):
            s1 = set(prev[1].split())
            s2 = set(sentence[1].split())
            actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
            if(actual_jaccard < 0.5): # if not similar
                processed_sentences.append(prev)
                prev = sentence
            else:
                print(prev)
                print(sentence)
                print(actual_jaccard)
                print('-------------------------------------------')
                
        s1 = set(prev[1].split())
        s2 = set(sentence[1].split())
        if(actual_jaccard < 0.5): # if not similar
            processed_sentences.append(prev)
        
        sorted_sentences = sorted(processed_sentences, key=lambda x:x[1])
        print(prev_len, len(sorted_sentences))
        
        if(prev_len == len(sorted_sentences)):
            break
        prev_len = len(sorted_sentences)
        
    return sorted_sentences


def get_first_topk_sentences(df):
    first_sentences = []
    topk_sentences = []
    topk_percent = 30
    for a,b in zip(df['context'], df['topk']):
        context = eval(str(a))
        topk = eval(str(b))
        k = int(len(topk)*(topk_percent/100))
        topk = topk[:k]
        
        first = []
        for item in topk:
            if(item[0] == 0): 
                first.append(item)
                
        if(len(first) == 0):
            first_sentences += [(0, context[0])]
            topk_sentences += topk
        else:
            first_sentences += first
            topk.remove(first[0])
            topk_sentences += topk
                
    print('before delete similar:', len(first_sentences), len(topk_sentences))
    first_sentences = delete_similar(first_sentences)
    topk_sentences = delete_similar(topk_sentences)
    print('after delete similar:', len(first_sentences), len(topk_sentences))
    return first_sentences, topk_sentences

def delete_repeat_str(user_text):
    
    text = user_text.split()
    x = len(text)
    comb = list(combinations(range(x), 2))
    sorted_comb = sorted(comb, key=lambda x: x[1]-x[0], reverse = True)
    for item in sorted_comb:
        start, end = item
        if(end-start <= len(sorted_comb) and end-start>1):
            find_str = ' '.join(text[start:end])
            rest_str = ' '.join(text[end:])
            idx = rest_str.find(find_str)
            if idx != -1:
                print('deleted :', find_str)
                ret = ' '.join(text[:end]) + ' ' + rest_str[:idx] + rest_str[idx+len(find_str)+1:]
                return ret
    return user_text


if __name__ == '__main__':
    topic_context_df = pd.read_csv('../paraphrase_topic20_pre.csv')
    first_sentences, topk_sentences = get_first_topk_sentences(topic_context_df)
    sum_result1 = summarize_first_sentences(first_sentences, tokenizer, model)
    sum_result2 = summarize_topk_sentences(topk_sentences, tokenizer, model)
    final_result = [v for i,v in [sum_result1] + sum_result2]
    final_sentences = [s if '.' in s[-2: ] else s+'. ' for i,s in enumerate(final_result)]
    new_final_sentences = []
    for s in final_sentences:
        new = delete_repet_str(s)
        new_final_sentences.append(delete_repeat_str(s))
        if(new != s):
            print(s)
            print('---------------------------------')
            print(new)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    print('======================= Final result =========================')
    print('\n'.join(new_final_sentences))