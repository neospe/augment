#!/usr/bin/env python

import os
import sys
import re
import uuid
import json
import html
import twint
import tweepy
import langid
import sqlite3
import requests
import datetime
from time import sleep
from tqdm import tqdm
from collections import Counter
import editdistance
from nltk.corpus import wordnet, stopwords
from germanet import load_germanet
from dataload.loader import *
from dataload.preproc import *


# PARAMETERS

# metadata
CONFIG_PATH = 'config.json'
BACKLOG_PATH = 'backlog.json'
SAMPLES_BACKLOG_PATH = 'debiasing-backlog.json'

# automation
interactive = False
default_action_rerun_config = 'n'  # in case of duplicate config, re-run? -> 'y' (yes) or any key (no)
default_action_duplicate_sample = 'd'  # in case of duplicate sample -> 'k' (keep) or 'd' (drop)

# generation settings
VARIANTS_PER_KEYWORD = 100
WORDS_DEBIAS_RANGE = 100  # no. of most negative connotations to balance
SYNONYMS_STOPLIST_DE = []  # optional: list of strings
SYNONYMS_STOPLIST_EN = []
DEBIASING_STOPLIST_DE = []  # optional: list of strings
DEBIASING_STOPLIST_EN = []

# data sources
MIOWORDCORPUS_PATH = '~/million_post_corpus/corpus.sqlite3'
WORDLIST_PATH_DE = ['de-keywords-from-your-domain.txt']
WORDLIST_PATH_EN = ['en-keywords-from-your-domain.txt']

# obsolete: twitter api
TWITTER_consumer_key=""
TWITTER_consumer_secret=""
TWITTER_access_token=""
TWITTER_access_token_secret=""

# external classification model
classification = False
API_HOST = None  # optional: string
API_KEY = None  # optional: string
API_INTERVAL = None  # optional: float
CLF_THRESHOLD = None  # optional: float


def simple_tsv_reader(path):
    f = open(path,'r', encoding='utf-8')
    data = []
    for line in f:
        entry = {}
        cols = line.split('\t')
        entry['text'] = cols[0]
        entry['label'] = cols[1].rstrip()
        data.append(entry)


def wordlist(path):
    wordlist = []
    with open(path,'r', encoding='utf-8') as f:
        for line in f:
            wordlist.append(line.rstrip())
    return list(set(wordlist))


def save_output(data_aug, outfile):
    print('* writing', len(data_aug), 'entries to', outfile)
    with open(outfile, 'w') as f:
        for entry in data_aug:
            text = entry['text']
            text = re.sub(r'[\r\n]+', ' |LBR| ', text)
            text = text.replace(' |LBR|  |LBR| ', ' |LBR| ')

            if len(entry) == 2:
                f.write(entry['text'] + '\t' + str(entry['label']) + '\n')
            elif len(entry) == 3:
                if not 'id' in entry:
                    f.write(entry['text'] + '\t' + str(entry['label']) + '\t' + str(entry['label2']) + '\n')
                else:
                    f.write(str(entry['id']) + '\t' + str(entry['text']) + '\t' + str(entry['label']) + '\n')
            elif len(entry) == 4:
                if not 'id' in entry:
                    f.write(entry['text'] + '\t' + str(entry['label']) + '\t' + str(entry['label2']) + '\t' + str(entry['label3']) + '\n')
                else:
                    f.write(str(entry['id']) + '\t' + entry['text'] + '\t' + str(entry['label']) + '\t' + str(entry['label2']) + '\n')
            elif len(entry) == 5:
                f.write(str(entry['id']) + '\t' + entry['text'] + '\t' + str(entry['label']) + '\t' + str(entry['label2']) + '\t' + str(entry['label3']) + '\n')
            else:
                print('ERROR: unexpected size of entry:', entry)


def sample_miowordcorpus(words, lang_code):
    """
    get samples from one million post corpus (sqlite)
    """
    data = []
    conn = sqlite3.connect(MIOWORDCORPUS_PATH)
    c = conn.cursor()

    for word, count in tqdm(words.items()):
        #c.execute('SELECT * FROM Posts WHERE Body LIKE "% '+word+' %" AND Status LIKE "online"')  # all
        c.execute('SELECT * FROM Posts WHERE Body LIKE "% '+word+' %" AND Status LIKE "online" AND id IN (SELECT id FROM Posts ORDER BY RANDOM() LIMIT '+str(count)+')')  # random sample
        i = 0
        for entry in c():
            if i == count:
                break

            headline = entry[6] or ''
            body = entry[7] or ''
            
            if headline == '':
                text = body.strip()
            elif body == '':
                text = headline.strip()
            else:
                text = headline.strip() + ' |LBR| ' + body.strip()

            # replace newlines
            text = re.sub(r'[\r\n]+', ' |LBR| ', text)
            text = text.replace(' |LBR|  |LBR| ', ' |LBR| ')

            # classify
            if classification:
                score = classify(text, 'de')
                sleep(API_INTERVAL)
                if score > CLF_THRESHOLD:
                    continue

            post = {}
            post['id'] = entry[0]
            post['text'] = text
            post['label'] = 0
            
            data.append(post)
            i += 1
        
        if i != count:
            print('%s: only %s items found' % (word, i))

    return data


def sample_twitter(words, lang_code):
    """
    get samples from twitter api
    """
    data = []

    try:
        auth = tweepy.OAuthHandler(TWITTER_consumer_key, TWITTER_consumer_secret)
        auth.set_access_token(TWITTER_access_token, TWITTER_access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)
    except tweepy.TweepError as e: 
        print(e.reason)

    for word, count in tqdm(words.items()):
        i = 0
        cursor = tweepy.Cursor(api.search, q=word, count=count, lang=lang_code, tweet_mode='extended').items()
        for tweet in cursor:
            if i == count:
                break

            if 'retweeted_status' in tweet._json:
                text = tweet._json['retweeted_status']['full_text']
            else:
                text = tweet.full_text

            # replace newlines
            text = re.sub(r'[\r\n]+', ' |LBR| ', text)
            text = text.replace(' |LBR|  |LBR| ', ' |LBR| ')

            # language detect
            if langid.classify(text.replace('|LBR|', ''))[0] != lang_code:
                continue

            # classify
            if classification:
                score = classify(text, lang_code)
                sleep(API_INTERVAL)
                if score > CLF_THRESHOLD:
                    continue

            post = {}
            post['id'] = tweet.id_str
            post['text'] = text
            post['label'] = 0
            
            data.append(post)
            i += 1
        
        if i != count:
            print('%s: only %s items found' % (word, i))

    return data


def sample_twint(words, lang_code):
    """
    get samples from twitter via twint
    """
    data = []

    for word, count in tqdm(words.items()):
        i = 0

        c = twint.Config()
        c.Search = word
        if count > 100:  # larger numbers sometimes don't get enough results
            c.Limit = count * 2
        else:
            c.Limit = count
        c.Lang = lang_code
        c.Hide_output = True
        c.Store_object = True
        twint.run.Search(c)
        
        for tweet in twint.output.tweets_list:
            if i == count:
                break

            text = tweet.tweet

            # replace newlines
            text = re.sub(r'[\r\n]+', ' |LBR| ', text)
            text = text.replace(' |LBR|  |LBR| ', ' |LBR| ')

            # language detect
            if langid.classify(text.replace('|LBR|', ''))[0] != lang_code:
                continue

            # classify
            if classification:
                score = classify(text, lang_code)
                sleep(API_INTERVAL)
                if score > CLF_THRESHOLD:
                    continue

            post = {}
            post['id'] = tweet.id
            post['text'] = text
            post['label'] = 'other'
            
            data.append(post)
            i += 1

        if i != count:
            print('%s: only %s items found' % (word, i))

    return data


def classify(text, lang_code):
    try: header = {'Authorization': 'Bearer ' + API_KEY}
    except: header = {}
    data = {'text': text}
    response = requests.post('https://'+API_HOST+'/'+lang_code+'/v1', json=data, headers=header)
    score = response.json()['scores'][0]

    return float(score)


def to_leet_speak(text):
    leet_dict = {
        'a': '4', 'b': '8', 'c': '<', 'e': '3', 'g': '6', 
        'i': '1', 'l': '|', 'o': '0', 's': '$', 't': '7'
    }
    return ''.join(leet_dict.get(char.lower(), char) for char in text)


def perturb(words):
    perturbations = {}
    
    for word in words:        
        p_tmp = []
        p_tmp.append(to_leet_speak(word.lower()))
        p_tmp.append(' '.join([char for char in word]))  # space between characters
        p_tmp.append('_'.join([char for char in word]))  # underscore
        p_tmp.append(word[0]+'*'*int((len(word)-1)*0.75))  # first character + '***'
        perturbations[word] = list(set(p_tmp))
    
    return perturbations


def augment_perturbations(data, config):
    """
    augment via perturbed keywords
    """

    # load keyword list
    if config['language_code'] == 'de':
        keyword_list = wordlist(WORDLIST_PATH_DE)
    elif config['language_code'] == 'en':
        keyword_list = wordlist(WORDLIST_PATH_EN)
    else:
        keyword_list = []

    data_aug = []
    keyword_list_done = {}
    got_to_next = False

    for i, entry in tqdm(enumerate(data)):

        # tokenize
        words = []
        for word in re.split(';|,|\'| |\.|-', entry['text']):
            if word != '':
                words.append(word)
        
        # match keywords using list
        keywords = []
        for word in words:
            for k_word in keyword_list:  # inefficient but we want to test for multiple conditions in the next line
                if k_word == word.lower():  #or (word.lower().startswith(k_word) and len(word) > len(k_word)+5):
                    keywords.append(word)

                    if k_word in keyword_list_done:
                        if keyword_list_done[k_word] <= VARIANTS_PER_KEYWORD:
                            keyword_list_done[k_word] +=1
                        else:
                            got_to_next = True
                            break
                    else:
                        keyword_list_done[k_word] = 1
        
        if got_to_next == True:
            got_to_next = False
            continue

        # replacement algorithm
        # goal: create as few quasi-identical examples as possible -> perform as many replacements as possible simultaneously, meaning all target words are replaced at the same time.
        # problem: the number of target words is completely variable + not every target word has the same number of candidate replacements.
        # solution: iterate through replacements for all target words simultaneously (with prior padding).

        if keywords != []:
            perturbations = perturb(keywords)

            # convert perturbations dict -> list of lists of tuples
            # {w1:[p1,p2,..], w2:[p1,p2,..],..} -> [[(w1, p1), (w1, p2),..], [(w2, p1), (w2, p2),..],..]
            list_of_lists = []
            for w, p in perturbations.items():
                list_of_tuples = []
                for p_ in p:
                    list_of_tuples.append((w, p_))
                list_of_lists.append(list_of_tuples)

            # determine maximum list length
            num_variants = 0
            for l in list_of_lists:
                if len(l) > num_variants:
                    num_variants = len(l)
            
            # padding lists to maximum length
            for i, l in enumerate(list_of_lists):
                if len(l) < num_variants:
                    last_elem = l[-1]
                    for i in range(num_variants-len(l)):
                        list_of_lists[i].append(last_elem)  # padding with last element
            
            # replace target words in one go
            for replacements in zip(*list_of_lists):  # unpacking list_of_lists
                new_entry = entry.copy()
                for r in replacements:
                    new_entry['text'] = new_entry['text'].replace(r[0], r[1])
                    data_aug.append(new_entry)

    return data_aug


def augment_synonyms(data, config):
    """
    augment via synonyms
    """
    data_aug = []

    # german version w/ germanet
    if config['language_code'] == 'de':
        gn = load_germanet()
        for i, entry in tqdm(enumerate(data)):
            for word in entry['text'].split():
                if len(word) > 3 or word[0].isupper():  # lookup only capitalized words or with length > 3 characters
                    lemma = gn.lemmatise(word)[0]
                    synsets = gn.synsets(lemma)
                    """
                    # exploration
                    if len(synsets) > 1:
                        print(synsets)

                    for syn in synsets:
                        if len(syn.lemmas) > 1:
                            print(syn)
                            print(syn.lemmas)
                            print('\n')
                    """
                    # simplest candidate selection: use only words with one sense/synset
                    if len(synsets) == 1 and lemma not in SYNONYMS_STOPLIST_DE:
                        done = []
                        for lem in synsets[0].lemmas:
                            syn_lemma = lem.orthForm
                            if syn_lemma != lemma and syn_lemma not in done and editdistance.eval(syn_lemma, lemma) > 2:  # no duplicates
                                new_entry = entry.copy()
                                new_entry['text'] = new_entry['text'].replace(word, syn_lemma)
                                data_aug.append(new_entry)
                                done.append(syn_lemma)

    # english version w/ wordnet
    elif config['language_code'] == 'en':
        for i, entry in tqdm(enumerate(data)):
            for word in entry['text'].split():
                if len(word) > 2:
                    syns = wordnet.synsets(word)
                    if len(syns) > 1:
                        word_syn = syns[0]

                        done = []
                        for syn in syns[1:]:
                            if word_syn.root_hypernyms() == syn.root_hypernyms():                            
                                similarity = word_syn.path_similarity(syn)
                                
                                if similarity < 1.0 and similarity > 0.3:
                                    syn_lemma = syn.lemmas()[0].name()
                                    """
                                    # exploration
                                    print(word_syn, '-->', word_syn.root_hypernyms())
                                    print(syn, '-->', syn.root_hypernyms())
                                    print('\n')
                                    print(similarity)
                                    """
                                    if syn_lemma != word and syn_lemma not in done and syn_lemma not in SYNONYMS_STOPLIST_EN and editdistance.eval(syn_lemma, word) > 2:  # no duplicates
                                        new_entry = entry.copy()
                                        new_entry['text'] = new_entry['text'].replace(word, syn_lemma)
                                        data_aug.append(new_entry)
                                        done.append(syn_lemma)

    else:
        sys.exit('%s not found.' % (config['language_code']))

    return data_aug


def augment_debiasing(data, config, uid):
    """
    augment via debiasing examples
    """
    
    # calculating differential term frequencies
    words_pos = []  
    words_neg = []  

    for entry in data:
        if entry['label'] > 0:   # positive class
            for word in re.split(';|,|\'| |\.|-', entry['text']):
                word = word.strip().lower()
                if word != '':
                    words_pos.append(word)
        else:                    # negative class
            for word in re.split(';|,|\'| |\.|-', entry['text']):
                word = word.strip().lower()
                if word != '':
                    words_neg.append(word)

    counter_pos = Counter(words_pos)
    #print(counter_pos.most_common(20), '\n')

    counter_neg = Counter(words_neg)
    #print(counter_neg.most_common(20), '\n')

    counter_neg.subtract(counter_pos)  # counter object always holds positive values
    #print(counter_neg.most_common(20), '\n')


    # load keyword list
    if config['language_code'] == 'de':
        keyword_list = wordlist(WORDLIST_PATH_DE)
    elif config['language_code'] == 'en':
        keyword_list = wordlist(WORDLIST_PATH_EN)
    else:
        keyword_list = []


    # search for neutral terms that are overrepresented in the positive class
    words = {}
    for word, count in counter_neg.most_common(WORDS_DEBIAS_RANGE):
        combined = False
        
        # ignore: keywords, function words, special characters
        # word[:-1] or [:-2] might match a singular form
        if config['language_code'] == 'de':
            if word in stopwords.words('german') or word in keyword_list or word[:-1] in keyword_list or word[:-2] in keyword_list or not word.isalpha() or word in DEBIASING_STOPLIST_DE:
                continue

        elif config['language_code'] == 'en':
            if word in stopwords.words('english') or word in keyword_list or word[:-1] in keyword_list or word[:-2] in keyword_list or not word.isalpha() or word in DEBIASING_STOPLIST_EN:
                continue

        # combine quasi-identical word forms, e.g. "merkel", "merkels"
        for k, v in words.items():
            if editdistance.eval(k, word) < 3:
                words[k] = v + count
                combined = True
                break

        # neues wort hinzufügen
        if combined == False:
            words[word] = count

    print('words to balance:', words)

    # search for examples for those terms that are neutral / negative class
    total_count = sum([v for _, v in words.items()])
    source = config['sample_source']

    if interactive is True:
        print('sampling %s items from %s. press any key to continue..' % (total_count, source))
        user_input = input()
    
    if source == 'twitter':
        data_aug = sample_twitter(words, config['language_code'])
    elif source == 'twint':
        data_aug = sample_twint(words, config['language_code'])
    elif source == 'miowordcorpus':
        data_aug = sample_miowordcorpus(words, config['language_code'])
    else:
        sys.exit('source %s not found.' % (source))


    # backlog for imported training examples
    # format: id__source: [uid, uid..]
    
    if os.path.exists(SAMPLES_BACKLOG_PATH):
        with open(SAMPLES_BACKLOG_PATH, 'r') as f:
            samples_backlog = json.load(f)
    else:
        print('%s not found. creating new backlog.' % (SAMPLES_BACKLOG_PATH))
        samples_backlog = {}

    posts_to_drop = []
    count_dupes = 0

    for i, post in enumerate(data_aug):
        if source == 'twint':
            post_id = str(post['id']) + '__twitter'
        else:
            post_id = str(post['id']) + '__' + source
        
        apply_action = 0
        if post_id in samples_backlog:
            count_dupes += 1
            jobs = samples_backlog[post_id]

            if interactive is True:
                if 'apply_to_all' not in config:
                    valid_input = 0
                    while valid_input == 0:
                        print('item %s from already sampled by job(s): %s' % (post_id, ', '.join(jobs)))
                        print('[k]eep or [d]rop?')
                        user_input = input()
                        if user_input == 'k':
                            samples_backlog[post_id].append(uid)
                            apply_action = 1
                            valid_input = 1
                        elif user_input == 'd':
                            posts_to_drop.append(i)
                            apply_action = 0
                            valid_input = 1
                    
                    valid_input = 0
                    while valid_input == 0:
                        print('apply to all? [y/n]')
                        user_input = input()
                        if user_input == 'y':
                            config['apply_to_all'] = 1
                            valid_input = 1
                        elif user_input == 'n':
                            config['apply_to_all'] = 0
                            valid_input = 1
                
                elif config['apply_to_all'] == 1:
                    if apply_action == 1:
                        samples_backlog[post_id].append(uid)
                    else:
                        posts_to_drop.append(i)
                
                elif config['apply_to_all'] == 0:
                    valid_input = 0
                    while valid_input == 0:
                        print('item %s from already sampled by job(s): %s' % (post_id, ', '.join(jobs)))
                        print('[k]eep or [d]rop?')
                        user_input = input()
                        if user_input == 'k':
                            samples_backlog[post_id].append(uid)
                            valid_input = 1
                        elif user_input == 'd':
                            posts_to_drop.append(i)
                            valid_input = 1
            
            else:  # not interactive
                if default_action_duplicate_sample == 'k':
                    samples_backlog[post_id].append(uid)
                else:
                    posts_to_drop.append(i)

        else:
            samples_backlog[post_id] = [uid]

    # write backlog
    with open(SAMPLES_BACKLOG_PATH, 'w') as f:
        json.dump(samples_backlog, f, indent=4)

    # drop selected posts
    for i in sorted(posts_to_drop, reverse=True):  # reversed so the indices keep stable
        del data_aug[i]
    
    print('%s dupes found, %s discarded.' % (count_dupes, len(posts_to_drop)))

    return data_aug


if __name__ == '__main__':

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config_file = json.load(f)
    else:
        sys.exit('%s not found.' % (CONFIG_PATH))

    if os.path.exists(BACKLOG_PATH):
        with open(BACKLOG_PATH, 'r') as f:
            backlog = json.load(f)
    else:
        print('%s not found. creating new backlog.' % (BACKLOG_PATH))
        backlog = {}
    
    for _, config in config_file.items():

        # check if job has been run before
        skip_aug = []
        rerun_aug = []
        for old_uid, old_config in backlog.items():
            if config['input_path'][0] == old_config['input_path'][0] and old_config['augmentations'] in config['augmentations'] and old_config['augmentations'] not in rerun_aug:
                
                if interactive is True:
                    print(config['dataset_name'], old_config['augmentations'])
                    print('has been run as job %s on %s and saved to: %s' % (old_uid, old_config['date'], old_config['output_path']))
                    print('to run again with the same settings, enter YES:')
                    user_input = input()
                    if user_input in ['YES', 'yes', 'Yes', 'y', 'Y']:
                        rerun_aug.append(old_config['augmentations'])
                    else:
                        skip_aug.append(old_config['augmentations'])
                else:
                    if default_action_rerun_config == 'y':
                        rerun_aug.append(old_config['augmentations'])
                    else:
                        skip_aug.append(old_config['augmentations'])
            
        # load data
        print('\n %s' % (config['dataset_name']))
        data = simple_tsv_reader(config['input_path'])
        print(len(data), 'items')

        # perform augmentation tasks
        for aug in config['augmentations']:
            if aug in skip_aug:
                continue

            print('*', aug)

            # generate uuid for this job
            uid = str(uuid.uuid4())

            if aug == 'perturbations':
                data_aug = augment_perturbations(data, config)
            elif aug == 'synonyms':
                data_aug = augment_synonyms(data, config)
            elif aug == 'debiasing':
                data_aug = augment_debiasing(data, config, uid)
            else:
                sys.exit('%s not found.' % (aug))

            # save augmented data
            OUTFILE = config['output_path']+'/'+uid+'.txt'
            save_output(data_aug, OUTFILE)

            # save config to backlog
            backlog[uid] = {
                'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dataset_name': config['dataset_name'],
                'input_path': config['input_path'],
                'augmentations': aug,
                'sample_source': config['sample_source'],
                'output_path': OUTFILE,
                'language_code': config['language_code']
            }

            with open(BACKLOG_PATH, 'w') as f:
                json.dump(backlog, f, indent=4)
