# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ETree
from os import walk

#DATA_LAF_DIR = "./annotation_data" #.laf.xml files
ANNOTATIONS_DIR = "./bolt_data/LDC2016E29_BOLT_LRL_Uzbek_Representative_Language_Pack_V1.0/data/annotation/entity_annotation/simple/"

#DATA_LTF_DIR = "./actual_data/" #.ltf.xml files
DATA_XML_DIR = "./bolt_data/LDC2016E29_BOLT_LRL_Uzbek_Representative_Language_Pack_V1.0/data/monolingual_text/ltf/"

TRAIN_CONLL_FILE = "./conll_generated_files/uzbek_train.conll"
DEV_CONLL_FILE = "./conll_generated_files/uzbek_dev.conll"
TEST_CONLL_FILE = "./conll_generated_files/uzbek_test.conll"
RAW_TEXT_CORPUS = "./conll_generated_files/uzbek_raw_text.txt"
TRAIN_SPLIT = 0.85
TEST_DEV_SPLIT = 0.5

if ANNOTATIONS_DIR[-1] is not '/':
    ANNOTATIONS_DIR += '/'

if DATA_XML_DIR[-1] is not '/':
    DATA_XML_DIR += '/'

def get_annotation_dict(annotations_dir):
    annotations_dict = {}
    # annotations_dict= {
    #                       <doc_id> : {
    #                                       <ner_entity> : {
    #                                                           (start_pos, end_pos, ner_type)
    #                                                      }
    #                                  }
    #                   }
    tot_incorrect_data = 0
    for [path, dirs, files] in walk(annotations_dir):
        for file in files:
            if '.xml' in file:
                file_path = path + file
                tree = ETree.parse(file_path)
                root = tree.getroot()
                for doc in root:
                    doc_id = doc.attrib['id']
                    lang = doc.attrib['lang']
                    #print(doc_id, lang)
                    for annotation in doc:
                        task = annotation.attrib['task']
                        #ner_type = annotation.attrib['type']
                        #print(task, ner_type)
                        for tag in annotation:
                            if tag.tag == "EXTENT":
                                extent = tag
                            elif tag.tag == "TAG":
                                ner_tag = tag

                        ner_type = ner_tag.text
                        #for extent in annotation:
                        ner_entities = extent.text.strip(' \t\r\n').split()
                        start_pos = int(extent.attrib['start_char'])
                        end_pos = int(extent.attrib['end_char'])
                        entity_start = start_pos
                        for ner_entity_index, ner_entity in enumerate(ner_entities):
                            #print(ner_entity)
                            #print(start_pos, end_pos)
                            if annotations_dict.get(doc_id, None) is None:
                                annotations_dict[doc_id] = {}
                                annotations_dict[doc_id][ner_entity] = {}
                            if annotations_dict[doc_id].get(ner_entity, None) is None:
                                annotations_dict[doc_id][ner_entity] = {}
                            if ner_entity_index == 0:
                                annotations_dict[doc_id][ner_entity][(entity_start, entity_start + len(ner_entity) - 1)] =  'B-'+ner_type
                            else:
                                annotations_dict[doc_id][ner_entity][
                                    (entity_start, entity_start + len(ner_entity) - 1)] = 'I-' + ner_type
                            #small sanity check
                            if ner_entity_index == len(ner_entities) - 1:
                                try:
                                    assert entity_start + len(ner_entity) - 1 == end_pos, "End pos doesn't match"
                                except:
                                    tot_incorrect_data += 1
                                    #print("We got a problem!")
                            entity_start += len(ner_entity) + 1
        break #To prevent recursive descent
    print("%d incorrect data found"%(tot_incorrect_data))
    return(annotations_dict)


def get_conll_from_xml_and_annotations(data_xml_dir, annotations_dict, train_conll_file, dev_conll_file, test_conll_file, raw_text_file, train_split, test_dev_split):
    # annotations_dict= {
    #                       <doc_id> : {
    #                                       <ner_entity> : {
    #                                                           (start_pos, end_pos, ner_type)
    #                                                      }
    #                                  }
    #                   }
    op_conll_corpus = []
    total_op_corpus = []
    for [path, dirs, files] in walk(data_xml_dir):
        for file in files:
            file_path = path + file
            tree = ETree.parse(file_path)
            root = tree.getroot()
            for doc in root:
                doc_id = doc.attrib['id']
                #print(doc_id)
                for text in doc:
                    for segment in text:
                        segment_start = segment.attrib['start_char']
                        segment_end = segment.attrib['end_char']
                        #print("In segment with range ", segment_start, segment_end)
                        annotation_list = []
                        for token in segment:
                            tag = token.tag
                            if tag == "ORIGINAL_TEXT":
                                continue
                            else:
                                word_token = token.text
                                pos = token.attrib['pos']
                                start_pos = int(token.attrib['start_char'])
                                end_pos = int(token.attrib['end_char'])
                                #Checking for sub-token entities
                                split_char = "ʼ".decode("UTF-8")
                                flag = 0
                                if len(word_token.split(split_char)) > 0:
                                    line_strs = []
                                    start_offset = 0
                                    for candidate_word in word_token.split(split_char):
                                        c_start_pos = start_pos + start_offset
                                        c_end_pos = c_start_pos + len(candidate_word) - 1
                                        start_offset = len(candidate_word) + 1 # CONFIRM THIS!
                                        try:
                                            line_strs.append(candidate_word + ' ' + pos + ' ' + annotations_dict[doc_id][candidate_word].get((c_start_pos, c_end_pos), 'O'))
                                            flag = 1
                                        except:
                                            line_strs.append(candidate_word + ' ' + pos + ' O')
                                    if flag == 1:
                                        annotation_list.extend(line_strs)
                                if flag == 0:
                                    try:
                                        line_str = word_token + ' ' + pos + ' ' + annotations_dict[doc_id][word_token].get((start_pos, end_pos), 'O')
                                    except:
                                        line_str = line_str = word_token + ' ' + pos + ' O'
                                    annotation_list.append(line_str)
                        sentence = '\n'.join(annotation_list)
                        total_op_corpus.extend([annotation.split()[0] for annotation in annotation_list])
                        op_conll_corpus.append(sentence)
        break
    train_end_index = int(len(op_conll_corpus) * train_split)
    train_conll_corpus = op_conll_corpus[:train_end_index]
    test_dev_conll_corpus = op_conll_corpus[train_end_index:]
    dev_end_index = int(len(test_dev_conll_corpus) * test_dev_split)
    dev_conll_corpus = test_dev_conll_corpus[:dev_end_index]
    test_conll_corpus = test_dev_conll_corpus[dev_end_index:]
    """
    with open(train_conll_file, 'w') as train_file:
        train_file.write('\n\n'.join(train_conll_corpus))
    with open(dev_conll_file, 'w') as dev_file:
        dev_file.write('\n\n'.join(dev_conll_corpus))
    with open(test_conll_file, 'w') as test_file:
        test_file.write('\n\n'.join(test_conll_corpus))
    """
    with open(train_conll_file, 'wb') as train_file:
        write_train_corp = [elt.encode('UTF-8') for elt in train_conll_corpus]
        # train_file.write('\n\n'.join(train_conll_corpus))
        train_file.write('\n\n'.join(write_train_corp))
    with open(dev_conll_file, 'wb') as dev_file:
        write_dev_corp = [elt.encode('UTF-8') for elt in dev_conll_corpus]
        # dev_file.write('\n\n'.join(dev_conll_corpus))
        dev_file.write('\n\n'.join(write_dev_corp))
    with open(test_conll_file, 'wb') as test_file:
        write_test_corp = [elt.encode('UTF-8') for elt in test_conll_corpus]
        # test_file.write('\n\n'.join(test_conll_corpus))
        test_file.write('\n\n'.join(write_test_corp))
    with open(raw_text_file, 'wb') as raw_file:
        write_raw_corp = " ".join([elt.encode('UTF-8') for elt in total_op_corpus])
        # test_file.write('\n\n'.join(test_conll_corpus))
        raw_file.write(write_raw_corp)


get_conll_from_xml_and_annotations(DATA_XML_DIR, get_annotation_dict(ANNOTATIONS_DIR), TRAIN_CONLL_FILE, DEV_CONLL_FILE,TEST_CONLL_FILE, RAW_TEXT_CORPUS, TRAIN_SPLIT, TEST_DEV_SPLIT)