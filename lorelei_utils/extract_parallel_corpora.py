import argparse
import xml.etree.ElementTree as ETree
import os

arg_parser = argparse.ArgumentParser()

# Args for Turkish

# arg_parser.add_argument("-trg", "--trg_lang", help="Target language", choices=["turkish", "uzbek", "english"], default="turkish",
#                         type=str)
#
# arg_parser.add_argument("-path", "--data_root_path", help="Root from where to start recursive search", default="./bolt_data/LDC2014E115_BOLT_LRL_Turkish_Representative_Language_Pack_V2.2/data/translation/from_eng/",
#                         type=str)
#
# arg_parser.add_argument("-op", "--output_file", help="Output file for the parallel corpus",
#                         default="parallel_corpus_tur_to_en.txt",
#                         type=str)

# arg_parser.add_argument("-trg", "--trg_lang", help="Target language", choices=["turkish", "uzbek", "english"],
#                         default="english",
#                         type=str)
#
# arg_parser.add_argument("-path", "--data_root_path", help="Root from where to start recursive search",
#                         default="./bolt_data/LDC2014E115_BOLT_LRL_Turkish_Representative_Language_Pack_V2.2/data/translation/from_tur/",
#                         type=str)
#
# arg_parser.add_argument("-op", "--output_file", help="Output file for the parallel corpus",
#                         default="parallel_corpus_en_to_tur.txt",
#                         type=str)

# Args for Uzbek

# arg_parser.add_argument("-trg", "--trg_lang", help="Target language", choices=["turkish", "uzbek", "english"],
#                         default="uzbek",
#                         type=str)
#
# arg_parser.add_argument("-path", "--data_root_path", help="Root from where to start recursive search",
#                         default="./bolt_data/LDC2016E29_BOLT_LRL_Uzbek_Representative_Language_Pack_V1.0/data"
#                                 "/translation/from_eng/",
#                         type=str)
#
# arg_parser.add_argument("-op", "--output_file", help="Output file for the parallel corpus",
#                         default="parallel_corpus_en_to_uzb.txt",
#                         type=str)

arg_parser.add_argument("-trg", "--trg_lang", help="Target language", choices=["turkish", "uzbek", "english"],
                        default="english",
                        type=str)

arg_parser.add_argument("-path", "--data_root_path", help="Root from where to start recursive search",
                        default="./bolt_data/LDC2016E29_BOLT_LRL_Uzbek_Representative_Language_Pack_V1.0/data"
                                "/translation/from_uzb/",
                        type=str)

arg_parser.add_argument("-op", "--output_file", help="Output file for the parallel corpus",
                        default="parallel_corpus_uzb_to_en.txt",
                        type=str)


# FIX THE OUTPUT ISSUE FOR PRINTING THE ENLGISH SIDE ON LEFT ALWAYS



arg_parser.add_argument("-tok", "--tokenize", help="Whether to use the DARPA tokenization",
                        default=1, choices=[1,0],
                        type=int)

args = arg_parser.parse_args()
print("Args used for this run:")
print(args)
trg_lang = args.trg_lang
data_path = args.data_root_path
OP_FILE = args.output_file
if args.tokenize == 1:
    tok_flag = "token"
else:
    tok_flag = "no_token"
if trg_lang == "turkish":
    trg_extension = "tur.ltf.xml"
elif trg_lang == "english":
    trg_extension = "eng.ltf.xml"
elif trg_lang == "uzbek":
    trg_extension = "uzb.ltf.xml"

src_extension = "ltf.xml"

trg_files = {}
src_files = {}


for path, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(trg_extension):
            trg_files[file.split(".")[0]] = os.path.abspath(os.path.join(path, file))
        elif file.endswith(src_extension):
            src_files[file.split(".")[0]] = os.path.abspath(os.path.join(path, file))

def get_segments_dict(file_path, mode="token"):
    segment_dict = {}
    tree = ETree.parse(file_path)
    root = tree.getroot()
    # print(file_path)
    for doc in root:
        doc_id = doc.attrib['id']
        for text in doc:
            for segment in text:
                segment_id = segment.attrib['id']
                segment_tokens = []
                for token in segment:
                    tag = token.tag
                    if tag == "ORIGINAL_TEXT":
                        if mode is not "token":
                            segment_dict[segment_id] = token.text
                            break
                    else:
                        if mode is not "token":
                            continue
                        word_token = token.text
                        segment_tokens.append(word_token)
                if mode is "token":
                    segment_dict[segment_id] = " ".join(segment_tokens)
    return(segment_dict)

def get_parallel_corpora(trg_files, src_files, OP_FILE, tokenize="token"):
    if tokenize == "token":
        print("Using BOLT tokenization")
    else:
        print("Not using BOLT tokenization")
    parallel_instances = []
    for trg_file_prefix in trg_files.keys():
        if src_files.get(trg_file_prefix, None) is None:
            continue
        src_file = src_files[trg_file_prefix]
        trg_file = trg_files[trg_file_prefix]
        src_seg_dict = get_segments_dict(src_file, mode=tokenize)
        trg_seg_dict = get_segments_dict(trg_file, mode=tokenize)
        for seg_id in src_seg_dict.keys():
            if trg_seg_dict.get(seg_id, None) is None:
                continue
            # Ensure English is always on the left side of the delimiter
            if trg_lang is "english":
                # parallel_instances.append([src_seg_dict[seg_id], trg_seg_dict[seg_id]])
                parallel_instances.append([trg_seg_dict[seg_id], src_seg_dict[seg_id]])
            else:
                parallel_instances.append([src_seg_dict[seg_id], trg_seg_dict[seg_id]])
                # parallel_instances.append([trg_seg_dict[seg_id], src_seg_dict[seg_id]])
    with open(OP_FILE, 'w') as op_file:
        op_file.write("\n".join([ " ||| ".join(parallel_instance) for parallel_instance in parallel_instances ]).encode("utf-8"))



get_parallel_corpora(trg_files, src_files, OP_FILE, tokenize=tok_flag)