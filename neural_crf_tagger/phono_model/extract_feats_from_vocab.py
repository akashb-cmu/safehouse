from my_utils import *
import os
sys.path.append("../epitran/epitran/bin/")
sys.path.append("../epitran/epitran/")
import vector

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-SV", "--src_vocab_file_prefix", help="Name prefix for vocabulary files with all source language"
                                                               "words for which features need to be extract",
                        type=str)
arg_parser.add_argument("-SD", "--src_vocab_dir", help="Directory that contains the source language vocabulary files",
                        type=str)
arg_parser.add_argument("-SC", "--src_lang_code", help="Epitran language code for the source language", choices=["aze-Cyrl",
                        "aze-Latn", "deu-Latn", "deu-Latn-np", "fra-Latn", "fra-Latn-np", "hau-Latn",
                        "ind-Latn", "jav-Latn", "kaz-Cyrl", "kaz-Latn", "kir-Arab", "kir-Cyrl", "kir-Latn", "nld-Latn",
                        "spa-Latn", "tuk-Cyrl", "tuk-Latn", "tur-Latn", "yor-Latn", "uig-Arab", "uzb-Cyrl", "uzb-Latn"],
                        default="tur-Latn",
                        type=str)
arg_parser.add_argument("-SS", "--src_lang_space", help="Epitran language space code for the source language",
                        choices=["tur-Latn-suf", "tur-Latn-nosuf", "uzb-Latn-suf", "spa-Latn", "nld-Latn", "deu-Latn"],
                        default="tur-Latn-suf",
                        type=str)
arg_parser.add_argument("-TV", "--trg_vocab_file_prefix",
                        help="Name prefix for vocabulary files with all target language words for which features need "
                             "to be extract",
                        type=str)
arg_parser.add_argument("-TD", "--trg_vocab_dir", help="Directory that contains the target language vocabulary files", type=str)
arg_parser.add_argument("-TC", "--trg_lang_code", help="Epitran language code for the source language",choices=["aze-Cyrl",
                        "aze-Latn", "deu-Latn", "deu-Latn-np", "fra-Latn", "fra-Latn-np", "hau-Latn",
                        "ind-Latn", "jav-Latn", "kaz-Cyrl", "kaz-Latn", "kir-Arab", "kir-Cyrl", "kir-Latn", "nld-Latn",
                        "spa-Latn", "tuk-Cyrl", "tuk-Latn", "tur-Latn", "yor-Latn", "uig-Arab", "uzb-Cyrl", "uzb-Latn"], default="uzb-Latn",
                        type=str)
arg_parser.add_argument("-TS", "--trg_lang_space", help="Epitran language space code for the target language",
                        choices=["tur-Latn-suf", "tur-Latn-nosuf", "uzb-Latn-suf", "spa-Latn", "nld-Latn", "deu-Latn"],
                        default="uzb-Latn-suf",
                        type=str)

args = arg_parser.parse_args()
print("Args used for this run:")
print(args)
src_vocab_file_prefix = args.src_vocab_file_prefix
src_vocab_dir = args.src_vocab_dir
src_lang_code = args.src_lang_code
src_lang_space = args.src_lang_space

trg_vocab_file_prefix = args.trg_vocab_file_prefix
trg_vocab_dir = args.trg_vocab_dir
trg_lang_code = args.trg_lang_code
trg_lang_space = args.trg_lang_space

word_categories = ['L', 'M', 'N', 'P', 'S', 'Z', 'C']




def read_all_vocabs(dir, file_prefix):
    vocab_set = set()
    for [path, dirs, files] in os.walk(dir):
        for file in files:
            if file_prefix in file:
                vocab_set = read_vocab_file(os.path.join(path, file), vocab_set=vocab_set)
    return(vocab_set)

src_vocab = read_all_vocabs(src_vocab_dir, src_vocab_file_prefix)
trg_vocab = read_all_vocabs(trg_vocab_dir, trg_vocab_file_prefix)

print(len(src_vocab))
print(len(trg_vocab))

src_epi = vector.VectorsWithIPASpace(src_lang_code, src_lang_space)
trg_epi = vector.VectorsWithIPASpace(trg_lang_code, trg_lang_space)


src_word = list(src_vocab)[0]
src_wvec = src_epi.word_to_segs(src_word, normpunc=True)

trg_word = list(trg_vocab)[0]
trg_wvec = trg_epi.word_to_segs(trg_word, normpunc=True)

word_phono_mat_dict, word_phono_char_vecs_mat_dict, word_ortho_char_vecs_mat_dict, word_cats_mat_dict, \
word_caps_vect_dict  = get_phono_vecs(src_vocab, src_epi, word_categories)

raw_input("Enter to end")