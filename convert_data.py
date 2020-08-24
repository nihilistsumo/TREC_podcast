import json
import os

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
                        if len(c.keys()) > 0:
                            episode += c['transcript']
                input_episode_ids.append(f)
                input_episodes.append(episode)
    with open(plain_output_file, 'w') as outfile:
        for e in input_episodes:
            plain_output_file.write(e+'\n')
    with open(output_file_id_file, 'w') as outid:
        json.dump(input_episode_ids, outid)