# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ETree
from os import walk
import argparse


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-laf", "--laf_data_dir", help="Folder with the NER annotations in laf format",
                        default="./bolt_data/LDC2016E29_BOLT_LRL_Uzbek_Representative_Language_Pack_V1.0/data/annotation/entity_annotation/simple/",
                        type=str)

arg_parser.add_argument("-ltf", "--ltf_data_dir", help="Folder with the raw text data",
                        default="./bolt_data/LDC2016E29_BOLT_LRL_Uzbek_Representative_Language_Pack_V1.0/data/monolingual_text/ltf/",
                        type=str)

arg_parser.add_argument("-op_pref", "--output_name_prefix", help="Prefix for the output file",
                        default="output_file_conll",
                        type=str)
arg_parser.add_argument("-exc_suf", "--exclude_suffixes", help="Whether to exclude suffixes from the NER span",
                        default=0, choices=[0,1],
                        type=int)
arg_parser.add_argument("-use_laf", "--use_laf", help="Whether LAF annotations are present and should be used",
                        default=1, choices=[0, 1],
                        type=int)
arg_parser.add_argument("-train_split", "--train_split", help="Percentage of data to use for training data",
                        default=0.9,
                        type=float)
arg_parser.add_argument("-test_dev_split", "--test_dev_split", help="Split within the non-training data into dev and test",
                        default=0.5,
                        type=float)


args = arg_parser.parse_args()
print("Args used for this run:")
print(args)



# ltf file has the actual corpus
# laf has the annotations


# DATA_LAF_DIR = "./annotation_data" #.laf.xml files
# For turkish
# DATA_LAF_DIR = "./bolt_data/LDC2014E115_BOLT_LRL_Turkish_Representative_Language_Pack_V2.2/data/annotation/entity_annotation/simple/"
# For Uzbek
# DATA_LAF_DIR = "./bolt_data/LDC2016E29_BOLT_LRL_Uzbek_Representative_Language_Pack_V1.0/data/annotation/entity_annotation/simple/"
DATA_LAF_DIR = args.laf_data_dir

# DATA_LTF_DIR = "./actual_data/" #.ltf.xml files
# For turkish
# DATA_LTF_DIR = "./bolt_data/LDC2014E115_BOLT_LRL_Turkish_Representative_Language_Pack_V2.2/data/monolingual_text/ltf/"
# For Uzbek
# DATA_LTF_DIR = "./bolt_data/LDC2016E29_BOLT_LRL_Uzbek_Representative_Language_Pack_V1.0/data/monolingual_text/ltf/"
DATA_LTF_DIR = args.ltf_data_dir
# CONLL_FILE_PREFIX = "./conll_generated_files/uzbek_to_adhi_conll"
CONLL_FILE_PREFIX = args.output_name_prefix

EXCLUDE_SUFFIXES_FLAG = False if args.exclude_suffixes is 0 else True

TRAIN_SPLIT = args.train_split
TEST_DEV_SPLIT = args.test_dev_split

if DATA_LAF_DIR[-1] is not '/':
    DATA_LAF_DIR += '/'

if DATA_LTF_DIR[-1] is not '/':
    DATA_LTF_DIR += '/'

USE_LAF = True if args.use_laf is 1 else False


def get_doc_tok_dict(data_ltf_dir): #, train_conll_file, dev_conll_file, test_conll_file, raw_text_file, train_split, test_dev_split):
    # annotations_dict= {
    #                       <doc_id> : {
    #                                       <ner_entity> : {
    #                                                           (start_pos, end_pos, ner_type)
    #                                                      }
    #                                  }
    #                   }
    # docstrings = {} # Maps a doc id to a continuous string representing its contents
    # doc_span_pos = {}
    doc_tok_dict = {}
    for [path, dirs, files] in walk(data_ltf_dir):
        files.sort() # Process in alphabetical order
        for file in files:
            file_path = path + file
            tree = ETree.parse(file_path)
            root = tree.getroot()
            print(file_path)
            for doc in root:
                doc_id = doc.attrib['id']
                # docstring = docstrings.get(doc_id, '')
                doc_tok_dict[doc_id] = {}
                for text in doc:
                    for segment in text:
                        segment_id = segment.attrib['id']
                        segment_start = segment.attrib['start_char']
                        segment_end = segment.attrib['end_char']
                        #print("In segment with range ", segment_start, segment_end)
                        annotation_list = []
                        for token in segment:
                            tag = token.tag
                            if tag == "ORIGINAL_TEXT":
                                continue
                            else:
                                #word_token = token.text.decode('utf8')
                                #word_token = token.text.encode('utf8')
                                word_token = token.text
                                pos = token.attrib['pos']
                                start_pos = int(token.attrib['start_char'])
                                end_pos = int(token.attrib['end_char'])
                                doc_tok_dict[doc_id][start_pos] = (word_token, end_pos, pos, segment_id)
                                #                 while start_pos > len(docstring):
                                #                     docstring += ' '
                                #                 docstring += word_token
                                #                 str_len = len(docstring)
                                #                 word_len = len(word_token)
                                #                 try:
                                #                     assert len(docstring) == end_pos + 1 and len(docstring) - len(word_token) == start_pos, "docstring and word don't align"
                                #                 except:
                                #                     print("We got a problem!")
                                #                 doc_span_pos[(doc_id, start_pos, end_pos)] = pos
                                # docstrings[doc_id] = docstrings.get(doc_id, '') + docstring

        break # prevent descending to subdirectories
    # return(docstrings, doc_span_pos)
    return doc_tok_dict


def get_turkish_conll_ner_instances(data_laf_dir, doc_tok_dict, respect_suffix_exclusion=False):
    annotations_dict = {}
    # annotations_dict= {
    #                       <doc_id> : {
    #                                       <seg_id> : [(token, tag)] ---> words in order
    #                                  }
    #                   }
    for [path, dirs, files] in walk(data_laf_dir):
        files.sort()
        for file in files:
            if '.xml' in file:
                file_path = path + file
                tree = ETree.parse(file_path)
                root = tree.getroot()
                for doc in root:
                    doc_id = doc.attrib['id']
                    lang = doc.attrib['lang']
                    # docstring = docstrings[doc_id]
                    if doc_tok_dict.get(doc_id, None) is not None: # some documents may only have annotations file
                        doc_toks_start_positions = doc_tok_dict[doc_id].keys()
                        doc_toks_start_positions.sort() # list of sorted token start positions
                        next_doc_tok_index = 0
                        annotations_dict[doc_id] = {}
                        for annotation in doc:
                            task = annotation.attrib['task']
                            ner_type = annotation.attrib['type']
                            #print(task, ner_type)
                            for extent in annotation:
                                start_pos = int(extent.attrib['start_char'])
                                end_pos = int(extent.attrib['end_char'])
                                while next_doc_tok_index <  len(doc_toks_start_positions) and doc_toks_start_positions[next_doc_tok_index] < start_pos:
                                    (tok, tok_end_pos, tok_POS_tag, tok_segment_id) = doc_tok_dict[doc_id][doc_toks_start_positions[next_doc_tok_index]]
                                    annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(tok_segment_id, []) + [(tok, 'O')]
                                    next_doc_tok_index += 1
                                flag = 0
                                while next_doc_tok_index <  len(doc_toks_start_positions) and doc_toks_start_positions[next_doc_tok_index] < end_pos:
                                    (tok, tok_end_pos, tok_POS_tag, tok_segment_id) = doc_tok_dict[doc_id][doc_toks_start_positions[next_doc_tok_index]]
                                    if flag == 0:
                                        if respect_suffix_exclusion:
                                            if tok_end_pos < end_pos:
                                                annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(
                                                    tok_segment_id, []) + [(tok, 'B-' + ner_type)]
                                            else:
                                                annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(
                                                    tok_segment_id, []) + [(tok[: end_pos - doc_toks_start_positions[next_doc_tok_index] + 1], 'B-' + ner_type)]
                                                if len(tok[end_pos - doc_toks_start_positions[
                                                    next_doc_tok_index] + 1:]
                                                       ) > 0:
                                                    annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(
                                                        tok_segment_id, []) + [(tok[end_pos - doc_toks_start_positions[
                                                        next_doc_tok_index] + 1:], 'O')]
                                        else:
                                            annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(tok_segment_id, []) + [(tok, 'B-' + ner_type)]
                                        flag = 1
                                    else:
                                        if respect_suffix_exclusion:
                                            if tok_end_pos < end_pos:
                                                annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(
                                                    tok_segment_id, []) + [(tok, 'I-' + ner_type)]
                                            else:
                                                annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(
                                                    tok_segment_id, []) + [(tok[: end_pos - doc_toks_start_positions[next_doc_tok_index] + 1], 'I-' + ner_type)]
                                                if len(tok[end_pos - doc_toks_start_positions[
                                                    next_doc_tok_index] + 1:]
                                                       ) > 0:
                                                    annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(
                                                        tok_segment_id, []) + [(tok[end_pos - doc_toks_start_positions[next_doc_tok_index] + 1 :], 'O')]
                                        else:
                                            annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(tok_segment_id, []) + [(tok, 'I-' + ner_type)]
                                    next_doc_tok_index += 1
                        if next_doc_tok_index < len(doc_toks_start_positions):
                            while next_doc_tok_index < len(doc_toks_start_positions):
                                (tok, tok_end_pos, tok_POS_tag, tok_segment_id) = doc_tok_dict[doc_id][doc_toks_start_positions[next_doc_tok_index]]
                                annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(tok_segment_id, []) + [(tok, 'O')]
                                next_doc_tok_index += 1

        break #To prevent recursive descent
    return(annotations_dict)

def get_uzbek_conll_ner_instances(data_laf_dir, doc_tok_dict, respect_suffix_exclusion=False):
    annotations_dict = {}
    # annotations_dict= {
    #                       <doc_id> : {
    #                                       <seg_id> : [(token, tag)] ---> words in order
    #                                  }
    #                   }
    for [path, dirs, files] in walk(data_laf_dir):
        files.sort()
        for file in files:
            if '.xml' in file:
                file_path = path + file
                tree = ETree.parse(file_path)
                root = tree.getroot()
                for doc in root:
                    doc_id = doc.attrib['id']
                    lang = doc.attrib['lang']
                    # docstring = docstrings[doc_id]
                    if doc_tok_dict.get(doc_id, None) is not None: # some documents may only have annotations file
                        doc_toks_start_positions = doc_tok_dict[doc_id].keys()
                        doc_toks_start_positions.sort() # list of sorted token start positions
                        next_doc_tok_index = 0
                        annotations_dict[doc_id] = {}
                        for annotation in doc:
                            task = annotation.attrib['task']
                            extent_list = []
                            ner_type_list = []
                            for sub_tag in annotation:
                                if sub_tag.tag == "EXTENT":
                                    extent_list.append(sub_tag)
                                elif sub_tag.tag == "TAG":
                                    ner_type_list.append(sub_tag.text)

                            #print(task, ner_type)
                            for extent_index, extent in enumerate(extent_list):
                                ner_type = ner_type_list[extent_index]
                                start_pos = int(extent.attrib['start_char'])
                                end_pos = int(extent.attrib['end_char'])
                                while next_doc_tok_index <  len(doc_toks_start_positions) and doc_toks_start_positions[next_doc_tok_index] < start_pos:
                                    (tok, tok_end_pos, tok_POS_tag, tok_segment_id) = doc_tok_dict[doc_id][doc_toks_start_positions[next_doc_tok_index]]
                                    annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(tok_segment_id, []) + [(tok, 'O')]
                                    next_doc_tok_index += 1
                                flag = 0
                                while next_doc_tok_index <  len(doc_toks_start_positions) and doc_toks_start_positions[next_doc_tok_index] < end_pos:
                                    (tok, tok_end_pos, tok_POS_tag, tok_segment_id) = doc_tok_dict[doc_id][doc_toks_start_positions[next_doc_tok_index]]
                                    if flag == 0:
                                        if respect_suffix_exclusion:
                                            if tok_end_pos < end_pos:
                                                annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(
                                                    tok_segment_id, []) + [(tok, 'B-' + ner_type)]
                                            else:
                                                annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(
                                                    tok_segment_id, []) + [(tok[: end_pos - doc_toks_start_positions[next_doc_tok_index] + 1], 'B-' + ner_type)]
                                                if len(tok[end_pos - doc_toks_start_positions[
                                                    next_doc_tok_index] + 1:]
                                                       ) > 0:
                                                    annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(
                                                        tok_segment_id, []) + [(tok[end_pos - doc_toks_start_positions[
                                                        next_doc_tok_index] + 1:], 'O')]
                                        else:
                                            annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(tok_segment_id, []) + [(tok, 'B-' + ner_type)]
                                        flag = 1
                                    else:
                                        if respect_suffix_exclusion:
                                            if tok_end_pos < end_pos:
                                                annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(
                                                    tok_segment_id, []) + [(tok, 'I-' + ner_type)]
                                            else:
                                                annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(
                                                    tok_segment_id, []) + [(tok[: end_pos - doc_toks_start_positions[next_doc_tok_index] + 1], 'I-' + ner_type)]
                                                if len(tok[end_pos - doc_toks_start_positions[
                                                    next_doc_tok_index] + 1:]
                                                       ) > 0:
                                                    annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(
                                                        tok_segment_id, []) + [(tok[end_pos - doc_toks_start_positions[next_doc_tok_index] + 1 :], 'O')]
                                        else:
                                            annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(tok_segment_id, []) + [(tok, 'I-' + ner_type)]
                                    next_doc_tok_index += 1
                        if next_doc_tok_index < len(doc_toks_start_positions):
                            while next_doc_tok_index < len(doc_toks_start_positions):
                                (tok, tok_end_pos, tok_POS_tag, tok_segment_id) = doc_tok_dict[doc_id][doc_toks_start_positions[next_doc_tok_index]]
                                annotations_dict[doc_id][tok_segment_id] = annotations_dict[doc_id].get(tok_segment_id, []) + [(tok, 'O')]
                                next_doc_tok_index += 1

        break #To prevent recursive descent
    return(annotations_dict)


def write_annotations_to_conll_file(conll_file_prefix, train_split, test_dev_split, annotations_dict):
    print_lines = []
    for doc_id in annotations_dict.keys():
        for segment_id in annotations_dict[doc_id]:
            print_lines.append("\n".join([tup[0] + "\t" + tup[1] for tup in annotations_dict[doc_id][segment_id]]))
    train_cutoff = int(train_split * len(print_lines))
    dev_cutoff = train_cutoff + int( (len(print_lines) - train_cutoff )*test_dev_split )

    with open(conll_file_prefix + "_train.conll", 'w') as train_file:
        train_file.write("\n\n".join(print_lines[:train_cutoff]).encode("utf8"))
    with open(conll_file_prefix + "_dev.conll", "w") as dev_file:
        dev_file.write("\n\n".join(print_lines[train_cutoff:dev_cutoff]).encode("utf8"))
    with open(conll_file_prefix + "_test.conll", 'w') as test_file:
        test_file.write("\n\n".join(print_lines[dev_cutoff:]).encode("utf8"))

# docstrings, doc_span_pos = get_doc_tok_dict(DATA_LTF_DIR)
doc_tok_dict = get_doc_tok_dict(DATA_LTF_DIR)

annotations_dict = get_turkish_conll_ner_instances(DATA_LAF_DIR, doc_tok_dict, EXCLUDE_SUFFIXES_FLAG) # Turkish method

# annotations_dict = get_uzbek_conll_ner_instances(DATA_LAF_DIR, doc_tok_dict, EXCLUDE_SUFFIXES_FLAG) # Uzbek method

write_annotations_to_conll_file(CONLL_FILE_PREFIX, TRAIN_SPLIT, TEST_DEV_SPLIT, annotations_dict)

raw_input("Enter to end!")