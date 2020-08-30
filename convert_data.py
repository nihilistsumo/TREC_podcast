import json
import os
from hashlib import sha1
import torch
from numpy.random import seed
seed(42)
torch.manual_seed(42)
import random
random.seed(42)

def convert_to_plain_data(root_dir, plain_output_file, output_file_id_file):
    input_episodes = []
    input_episode_ids = []
    for dir_name, subdirs, files in os.walk(root_dir):
        if len(files) > 0:
            for f in files:
                with open(os.path.join(dir_name, f), 'r') as infile:
                    input_data = json.load(infile)
                episode = ''
                for chunk in input_data['results']:
                    for c in chunk['alternatives']:
                        if 'transcript' in c.keys():
                            episode += c['transcript']
                input_episode_ids.append(f)
                input_episodes.append(episode)
    with open(plain_output_file, 'w') as outfile:
        for e in input_episodes:
            outfile.write(e+'\n')
    with open(output_file_id_file, 'w') as outid:
        json.dump(input_episode_ids, outid)

def produce_gan_podcast_run(result_file, episode_id_file, root_dir, output_dir):
    with open(episode_id_file, 'r') as ei:
        episodes = json.load(ei)
    with open(result_file, 'r') as rf:
        summaries = rf.readlines()
    assert len(episodes) == len(summaries)

def split_plain_input_text(input_text, outdir):
    lines = []
    with open(input_text, 'r') as pi:
        for l in pi:
            lines.append(l)
    line_words = []
    for l in lines:
        line_words.append(l.split())
    chunks = []
    for l in lines:
        chunks.append([])
    chunk_size = 1000
    iterations = 10
    for i in range(iterations):
        for j in range(len(lines)):
            words = line_words[j]
            if i * chunk_size < len(words):
                s = ' '.join(words[i * chunk_size:(i + 1) * chunk_size])
                chunks[j].append(s)
            else:
                chunks[j].append('.')
    for i in range(len(chunks[0])):
        with open(outdir + '/input_split_' + str(i + 1), 'w') as pi:
            for l in chunks:
                pi.write(l[i].strip() + '\n')

def convert_qrels_to_secid_dict(qrels_file, secid_dict_output):
    qrels = {}
    with open(qrels_file, 'r') as qf:
        for l in qf:
            q = l.split(' ')[0]
            p = l.split(' ')[2]
            if q in qrels.keys():
                qrels[q].append(p)
            else:
                qrels[q] = [p]
    print('Total '+str(len(qrels))+' sections')
    qrels_secid = {}
    for q in qrels.keys():
        qtext = q.split(':')[1].replace('/', ' ').replace('%20', ' ')
        qhash = sha1(str.encode(qtext)).hexdigest()
        qrels_secid['SECID:' + qhash] = {'paras': qrels[q], 'qtext': qtext}
        if len(qrels_secid) % 100 == 0:
            print(len(qrels_secid))
    with open(secid_dict_output, 'w') as out:
        json.dump(qrels_secid, out)