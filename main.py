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
import argparse

def run_summary(root_dir, output_dir, sorted_summary_path, max_summary_lines):
    with open(sorted_summary_path, 'r') as summ:
        sorted_summary = json.load(summ)
    print(root_dir)
    for dir_name, subdirs, files in os.walk(root_dir):
        for d in subdirs:
            os.makedirs(os.path.join(output_dir + '/' + dir_name[len(root_dir)+1:] +'/' + d))
        if len(files) > 0:
            for f in files:
                ep_id = f.split('.')[0]
                with open(os.path.join(output_dir + '/' + dir_name[len(root_dir)+1:] + '/' + ep_id + '_summary.txt'),
                          'w') as out:
                    summary = '. '.join(sorted_summary[f][:max_summary_lines])
                    out.write(summary)

def generate_sorted_summary_lines(summary_lines_dict, summary_vec_gen_model, out_file, max_seq_len=10, emb_vec_size=768,
                     embed_model_name='bert-base-wikipedia-sections-mean-tokens'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(summary_lines_dict, 'r') as sd:
        summary_lines = json.load(sd)
    m = SummaryEmbedGen(emb_vec_size, max_seq_len)
    m.load_state_dict(torch.load(summary_vec_gen_model))
    m.to(device)
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
    X = torch.tensor(X).float().to(device)
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
    parser = argparse.ArgumentParser(description='Generate summary files in TREC2020 Podcast track format')
    parser.add_argument('-id', '--input_dir', help='Path to TREC2020 podcast summarization task input directory',
                        default='/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/spotify-podcasts-2020/podcasts-transcripts-summarization-testset')
    parser.add_argument('-od', '--output_dir', help='Path to summary result output directory',
                        default='/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/spotify-podcasts-2020/output')
    parser.add_argument('-si', '--sorted_summary_dict', help='Path to sorted summary dict',
                        default='/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/summary_data/sorted_podcast_summary_seq_15_seq_len_1000_by1train_gensum.json')
    parser.add_argument('-ml', '--max_summary_len', type=int, default=3, help='Max summary lines count')
    args = parser.parse_args()

    run_summary(args.input_dir, args.output_dir, args.sorted_summary_dict, args.max_summary_len)

if __name__ == '__main__':
    main()
