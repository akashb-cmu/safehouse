import xml.etree.ElementTree as ETree
from os import walk

WIKI_DIR = './wiki_data'
OUTPUT_FILE = './processed_wiki_data.txt'

def get_raw_text(wiki_data_dir, output_file):
    if wiki_data_dir[-1] is not '/':
        wiki_data_dir += '/'
    corpus  = []
    puncts = ['.', '!', '?']
    for [path, dirs, files] in walk(wiki_data_dir):
        for file in files:
            if '.xml' in file:
                with open(path + '/' + file, 'r') as ip_file:
                    for line in ip_file:
                        line = line.strip(' \t\r\n')
                        if len(line) > 0 and line[0] is not '<':
                            if line[-1] not in puncts:
                                line += '.'
                            corpus.append(line)
    with open(output_file, 'w') as op_file:
        op_file.write(' '.join(corpus))
    return(corpus)

get_raw_text(WIKI_DIR, OUTPUT_FILE)