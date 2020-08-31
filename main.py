import os
import torch
import numpy as np
from numpy.random import seed
seed(42)
torch.manual_seed(42)
import random
random.seed(42)
import json
from summary_vec_gen import SummaryEmbedGen
from sentence_transformers import SentenceTransformer

def run_summary(root_dir, output_dir):
    print(root_dir)
    for dir_name, subdirs, files in os.walk(root_dir):
        for d in subdirs:
            os.makedirs(os.path.join(output_dir + '/' + dir_name[len(root_dir)+1:] +'/' + d))
        if len(files) > 0:
            for f in files:
                with open(os.path.join(output_dir + '/' + dir_name[len(root_dir)+1:] + '/' + f), 'w') as out:
                    out.write('file written')

def generate_sorted_summary_lines(summary_lines_dict, summary_vec_gen_model, out_file, max_seq_len=10, emb_vec_size=768,
                     embed_model_name='bert-base-wikipedia-sections-mean-tokens'):
    with open(summary_lines_dict, 'r') as sd:
        summary_lines = json.load(sd)
    m = SummaryEmbedGen(emb_vec_size, max_seq_len).cuda()
    m.load_state_dict(torch.load(summary_vec_gen_model))
    m.eval()
    emb_model = SentenceTransformer(embed_model_name)
    X = []
    ep_ids = []
    for ep_id in summary_lines.keys():
        ep_ids.append(ep_id)
        summary_list = summary_lines[ep_id]['summary_lines']
        summary_emb_vecs = emb_model.encode(summary_list, show_progress_bar=True)
        if summary_emb_vecs.shape[0] < max_seq_len:
            summary_emb_vecs = np.vstack((summary_emb_vecs,
                                          np.zeros((max_seq_len - summary_emb_vecs.shape[0], emb_vec_size))))
        X.append(summary_emb_vecs)
    X = torch.tensor(X).cuda()
    inferred_summary_vec = m(X).detach().cpu().numpy()
    X = X.detach().cpu().numpy()
    summary_sim_matrix = np.array([np.dot(X[i], inferred_summary_vec[i])/
                                   (np.linalg.norm(X[i])*np.linalg.norm(inferred_summary_vec[i]))
                                   for i in range(X.shape[0])])
    summary_sort_indices = np.argsort(summary_sim_matrix)
    sorted_summaries = {}
    for i in range(len(ep_ids)):
        ep_id = ep_ids[i]
        summary_list = summary_lines[ep_id]['summary_lines']
        sorted_summary_list = []
        for j in reversed(list(summary_sort_indices[i])):
            if j < len(summary_list):
                sorted_summary_list.append(summary_list[j])
        sorted_summaries[ep_id] = sorted_summary_list
    with open(out_file, 'w') as out:
        json.dump(sorted_summaries, out)



def main():
    run_summary('/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/spotify-podcasts-2020/podcasts-transcripts-summarization-testset',
                '/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/spotify-podcasts-2020/output')

if __name__ == '__main__':
    main()
