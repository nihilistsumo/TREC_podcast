import os

def run_summary(root_dir, output_dir):
    print(root_dir)
    for dir_name, subdirs, files in os.walk(root_dir):
        for d in subdirs:
            os.makedirs(os.path.join(output_dir + '/' + dir_name[len(root_dir)+1:] +'/' + d))
        if len(files) > 0:
            for f in files:
                with open(os.path.join(output_dir + '/' + dir_name[len(root_dir)+1:] + '/' + f), 'w') as out:
                    out.write('file written')

def main():
    run_summary('/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/spotify-podcasts-2020/podcasts-transcripts-summarization-testset',
                '/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/spotify-podcasts-2020/output')

if __name__ == '__main__':
    main()
