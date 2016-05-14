"""
This script obtains a dictionary between a foreign language and a source language given bi-directional model 1 parameters
for the language pair
"""
import argparse
from Model1_EM import *


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-fwd", "--forward_model", help="trg_given_src IBM model 1",
                        default="./models/eng_given_tur_model1.model",
                        type=str)

arg_parser.add_argument("-bwd", "--backward_model", help="src_given_trg IBM model 1",
                        default="./models/tur_given_eng_model1.model",
                        type=str)

arg_parser.add_argument("-src", "--src_lang", help="Source language (RHS of |||)",
                        choices=["english", "german", "turkish", "uzbek"],
                        default="turkish",
                        type=str)

arg_parser.add_argument("-trg", "--foreign_lang", help="Foreign language (LHS of |||)",
                        choices=["english", "german", "turkish", "uzbek"],
                        default="english",
                        type=str)

arg_parser.add_argument("-ip", "--ip_file", help="Input file with the original parallel data",
                        default="./data/all_parallel_tur_data.txt",
                        type=str)

arg_parser.add_argument("-dict", "--dict_file", help="Output dictionary file",
                        default="./eng_tur_dict.txt",
                        type=str)

arg_parser.add_argument("-thresh", "--probab_threshold", help="Threshold to consider a word pair a translation",
                        default=0.1,
                        type=float)
# IP FILE should have parllel instances of the form <source sentence> ||| <target sentence>


args = arg_parser.parse_args()
print("Args used for this run:")
print(args)

FWD_MODEL = args.forward_model
BWD_MODEL = args.backward_model
SRC_LANG = args.src_lang
TRG_LANG = args.foreign_lang
IP_FILE = args.ip_file
THRESH = args.probab_threshold
DICT_FILE = args.dict_file

print("Loading model")

fwd_model = Model1(parameter_file = FWD_MODEL, foreign_language = TRG_LANG, source_language = SRC_LANG)
bwd_model = Model1(parameter_file = BWD_MODEL, foreign_language = TRG_LANG, source_language = SRC_LANG)

print("Loaded models. Now obtaining vocabs")

def get_bitext_vocabs(parallel_ip_file):
    src_vocab = set()
    trg_vocab = set()
    with open(parallel_ip_file, "r") as ip_file:
        for line in ip_file:
            line = line.strip()
            if len(line) > 0:
                try:
                    [foreign, source] = [sent.decode('utf-8').strip(' \t\r\n').encode("utf-8") for sent in line.split('|||')]
                except:
                    print(line)
                    raw_input("Enter to continue")
                source = source.split()
                foreign = foreign.split()
                src_vocab.update(source)
                trg_vocab.update(foreign)
    return(src_vocab, trg_vocab)

def get_model_vocabs(model, threshold):
    l1_stem_vocab = set()
    l2_stem_vocab = set()
    for key in model.get_params().keys():
        if model.get_params()[key] > threshold:
            (l1_stem, l2_stem) = key
            l1_stem_vocab.add(l1_stem)
            l2_stem_vocab.add(l2_stem)
    return(l1_stem_vocab, l2_stem_vocab)



trg_model_vocab1, src_model_vocab1 = get_model_vocabs(fwd_model, THRESH)
src_model_vocab2, trg_model_vocab2 = get_model_vocabs(bwd_model, THRESH)

trg_model_vocab = trg_model_vocab1.intersection(trg_model_vocab2)
src_model_vocab = src_model_vocab1.intersection(src_model_vocab2)

print(len(trg_model_vocab), len(src_model_vocab))

src_vocab, trg_vocab = get_bitext_vocabs(IP_FILE)

print(len(trg_vocab), len(src_vocab))

print("Obtaining dictionaries at suffix level")

model_bilingual_dict = {}


# for src_word in src_vocab:
for src_stem in src_model_vocab:
    # for trg_word in trg_vocab:
    for trg_stem in trg_model_vocab:
        # src_stem = fwd_model.get_stem(src_word, SRC_LANG)
        # trg_stem = fwd_model.get_stem(trg_word, TRG_LANG)
        bidir_probab = fwd_model.get_translation_prob(trg_stem, src_stem) * bwd_model.get_translation_prob(src_stem, trg_stem)
        if bidir_probab > THRESH:
            model_bilingual_dict[trg_stem] = set([src_stem]).union(model_bilingual_dict.get(trg_stem, set()))

print("Converting to word level dictionary")

bilingual_dict = {}

suffix_to_word_map = {}

for src_word in src_vocab:
    suffix_to_word_map[fwd_model.get_stem(src_word, SRC_LANG)] = set([src_word]).union(suffix_to_word_map.get(fwd_model.get_stem(src_word, SRC_LANG), set()))

def suffix_to_word(suffix_set, suffix_to_word_map):
    ret_set = set()
    for suffix in suffix_set:
        if suffix_to_word_map.get(suffix, None) is not None:
            ret_set.update(list(suffix_to_word_map[suffix]))
    return(ret_set)


for trg_word in trg_vocab:
    trg_stem = fwd_model.get_stem(trg_word, TRG_LANG)
    if model_bilingual_dict.get(trg_stem, None) is not None:
        bilingual_dict[trg_word] = suffix_to_word(model_bilingual_dict[trg_stem], suffix_to_word_map)

print("Obtained dictionary")

print(len(bilingual_dict))

print("Writing out dictionary to file")

with open(DICT_FILE, 'w') as op_file:
    for key in bilingual_dict.keys():
        for translation in bilingual_dict[key]:
            if len(translation.strip("\t\r\n ")) > 0:
                op_file.write(str(key) + " ||| " + translation + "\n")