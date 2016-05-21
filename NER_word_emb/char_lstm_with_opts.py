# -*- coding: utf-8 -*-
from keras.models import Graph
from keras.layers.core import Dense, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.regularizers import l1,l2
from keras.optimizers import RMSprop
import numpy as np
import pickle
import sys
import pylab as pl
import argparse

sys.path.append("./epitran/epitran/bin/")
sys.path.append("./epitran/epitran/")
import word2pfvectors
import _epitran as epitran

# word = 'Özturanʼın'.decode("utf-8")
# word = "Özturan.12ın".decode("utf-8")
# ret_val = epi.word_to_pfvector(word)
# raw_input("Enter to end")

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-s", "--source", help="Source language", choices=["turkish", "uzbek"], default="turkish",
                    type=str)
arg_parser.add_argument("-t", "--target", help="Target language", choices=["turkish", "uzbek"], default="uzbek",
                        type=str)
arg_parser.add_argument("-tsplit", "--test_split", help="Percentage of data to use as test", default=0.1,
                        type=float)
arg_parser.add_argument("-nbatch", "--batch_size", help="Number of samples to include in a batch", default=64,
                        type=int)
arg_parser.add_argument("-nepoch", "--num_epochs", help="Number of epochs to run for", default=100,
                        type=int)
arg_parser.add_argument("-edim", "--embed_dim", help="Embeddings dimension", default=60,
                        type=int)
arg_parser.add_argument("-ldim", "--hidden_dim_lstm", help="LSTM hidden layer dimension", default=100,
                        type=int)
arg_parser.add_argument("-pre_dim", "--pre_predict_dim", help="Dimension of layer before final prediction", default=150,
                        type=int)
arg_parser.add_argument("-lreg", "--lstm_reg_weight", help="Regularization to apply for LSTM weights", default=0.01,
                        type=float)
arg_parser.add_argument("-pre_reg", "--pre_predict_reg_weight", help="Regularization to apply for pre prediction layer weights",
                        default=0.01,
                        type=float)
arg_parser.add_argument("-pred_reg", "--predict_reg_weight", help="Regularization to apply for final prediction layer weights",
                        default=0.01,
                        type=float)
arg_parser.add_argument("-use_src", "--use_src",
                        help="Whether to use the source language phonotactic and orthographic vocabulary",
                        default=0, choices=[0,1],
                        type=int)
arg_parser.add_argument("-use_trg", "--use_trg",
                        help="Whether to use the target language phonotactic and orthographic vocabulary",
                        default=0, choices=[0,1],
                        type=int)
arg_parser.add_argument("-use_ortho", "--use_ortho",
                        help="Whether to use orhtographic features",
                        default=0, choices=[0,1],
                        type=int)
arg_parser.add_argument("-use_phono", "--use_phono",
                        help="Whether to use phonotactic features",
                        default=0, choices=[0,1],
                        type=int)
arg_parser.add_argument("-use_wvecs", "--use_word_vectors",
                        help="Whether to use word vectors",
                        default=0, choices=[0, 1],
                        type=int)
arg_parser.add_argument("-w_len", "--word_length_threshold", help="Cutoff in terms of word length", default=40,
                        type=int)
arg_parser.add_argument("-src_wvecs", "--source_word_vecs", help="Word vectors for the source language",
                        default=None,
                        type=str)
arg_parser.add_argument("-trg_wvecs", "--target_word_vecs", help="Word vectors for the target language",
                        default=None,
                        type=str)
arg_parser.add_argument("-mname", "--model_name", help="Name of the model", default="model_stur_tuzb_phono_ortho_src_only",
                        type=str)
arg_parser.add_argument("-use_caps", "--use_capitalization_feature",
                        help="Whether to use capitalization features",
                        default=1, choices=[0, 1],
                        type=int)
arg_parser.add_argument("-use_cats", "--use_word_categories",
                        help="Whether to use word category features",
                        default=1, choices=[0, 1],
                        type=int)

args = arg_parser.parse_args()
print("Args used for this run:")
print(args)
src_lang = args.source
trg_lang = args.target

USE_WORD_VECS = False if args.use_word_vectors is 0 else True
SRC_WVECS = args.source_word_vecs
TRG_WVECS = args.target_word_vecs

if USE_WORD_VECS:
    assert None not in [SRC_WVECS, TRG_WVECS], "Word vector files not specified"

EMBEDDING_DIM = args.embed_dim
LSTM_OUTPUT_DIM = args.hidden_dim_lstm
PRE_PREDICT_DIM = args.pre_predict_dim
LSTM_REG_WEIGHT = args.lstm_reg_weight
PRE_PRED_REG = args.pre_predict_reg_weight
PRED_REG = args.predict_reg_weight

USE_SRC = args.use_src
USE_TRG = args.use_trg

USE_ORTHO = args.use_ortho

USE_PHONO = args.use_phono

WORD_LEN_THRESHOLD = args.word_length_threshold

TEST_SPLIT = args.test_split

N_BATCH = args.batch_size
N_EPOCH = args.num_epochs

USE_CAPS = True if args.use_capitalization_feature is 1 else False
USE_CATS = True if args.use_word_categories is 1 else False

assert src_lang != trg_lang, "Source and target languages must be different"

if src_lang == "turkish":
    TRAIN_CONLL = "./conll_data/turkish_conll_train.conll"
    TEST_CONLL = "./conll_data/turkish_conll_test.conll"
    DEV_CONLL = "./conll_data/turkish_conll_dev.conll"
    SRC_LANG = "tur-Latn"
elif src_lang == "uzbek":
    TRAIN_CONLL = "./conll_data/uzbek_conll_train.conll"
    TEST_CONLL = "./conll_data/uzbek_conll_test.conll"
    DEV_CONLL = "./conll_data/uzbek_conll_dev.conll"
    SRC_LANG = "uzb-Latn"

if trg_lang == "uzbek":
    TARGET_TRAIN_CONLL = "./conll_data/uzbek_conll_train.conll"
    TARGET_TEST_CONLL = "./conll_data/uzbek_conll_test.conll"
    TARGET_DEV_CONLL = "./conll_data/uzbek_conll_dev.conll"
    TRG_LANG = "uzb-Latn"
elif trg_lang == "turkish":
    TARGET_TRAIN_CONLL = "./conll_data/turkish_conll_train.conll"
    TARGET_TEST_CONLL = "./conll_data/turkish_conll_test.conll"
    TARGET_DEV_CONLL = "./conll_data/turkish_conll_dev.conll"
    TRG_LANG = "tur-Latn"

MODEL_NAME_PREFIX = args.model_name + ".model"
model_prefix = MODEL_NAME_PREFIX + ".configs"
print("Writing configs for model")
pickle.dump(args, open(model_prefix, 'wb'))

def get_words_charset_and_ner_sets(char_vocab, ner_sets_dict=None, non_ner_set=None, file=None, word_len_dict={}, ignore_words=False, threshold=sys.maxint):
    assert file is not None, "File not specified"
    if ignore_words == False:
        assert ner_sets_dict is not None and non_ner_set is not None, "Can't create vocabs with null args"
    with open(file, 'r') as ip_file:
        for line in ip_file:
            line = line.strip(' \t\r\n')
            if len(line) == 0:
                continue
            word, tag = line.split("\t")
            word = word.decode('utf8')
            word_len_dict[len(word)] = word_len_dict.get(len(word), 0.0) + 1
            word = word[len(word) - threshold : ] # Taking suffixes to capture the suffixal morphology
            if not ignore_words:
                if 'B-' in tag or 'I-' in tag:
                    ner_sets_dict[tag[2:]] = set([word]).union(ner_sets_dict.get(tag[2:], set()))
                    if word in non_ner_set:
                        non_ner_set.remove(word)
                elif not any(word in ner_set for ner_set in ner_sets_dict.values()) : # MAKE NERs and non-NERs clearly separable
                    non_ner_set.add(word)
            char_vocab.update(list(word)) # Should this be done AFTER word truncation?

    return(char_vocab, ner_sets_dict, non_ner_set, word_len_dict)

def get_ipa_ids(word, epi):
    ipa_vecs = epi.word_to_pfvector(word)
    ipa_ids = []
    for vec in ipa_vecs:
        (character_category,
         is_upper,
         phonetic_form,
         id_in_unicode_ipa_space,
         phonological_feature_vector) = vec
        ipa_ids.append(id_in_unicode_ipa_space)
    return(ipa_ids)

def get_ipa_charset_and_ner_sets(ipa_vocab, ner_sets_dict, non_ner_set, file, epi, ignore_words=False, threshold=sys.maxint):
    if ignore_words == False:
        assert ner_sets_dict is not None and non_ner_set is not None, "Can't create vocabs with null args"
    with open(file, 'r') as ip_file:
        for line in ip_file:
            line = line.strip(' \t\r\n')
            if len(line) == 0:
                continue
            word, tag = line.split("\t")
            word = word.decode('utf8')
            word = word[len(word) - threshold : ]
            if not ignore_words:
                if 'B-' in tag or 'I-' in tag:
                    ner_sets_dict[tag[2:]] = set([word]).union(ner_sets_dict.get(tag[2:], set()))
                    if word in non_ner_set:
                        non_ner_set.remove(word)
                elif not any(word in ner_set for ner_set in ner_sets_dict.values()) : # MAKE NERs and non-NERs clearly separable
                    non_ner_set.add(word)
                ipa_vocab.update(get_ipa_ids(word, epi))
    return(ipa_vocab, ner_sets_dict, non_ner_set)

def get_one_hot_char(n_chars, char_id):
    assert char_id <= n_chars, "Invalid char id"
    char_vect = [0 for i in range(n_chars + 1)]
    char_vect[char_id] = 1
    return char_vect

# def get_instance_from_word(word, epi, char_vocab, word_category_sets):
def get_instance_from_word(word, epi, char_vocab, word_category_sets, use_caps=True):
    vecs = epi.word_to_pfvector(word)
    ret_vecs = []
    n_chars = len(char_vocab.keys())
    for vec in vecs:
        (character_category,
         is_upper,
         phonetic_form,
         id_in_unicode_ipa_space,
         phonological_feature_vector) = vec
        c_vec = get_one_hot_char(n_chars, char_vocab[id_in_unicode_ipa_space])
        c_vec.extend(phonological_feature_vector)
        if use_caps:
            c_vec.append(is_upper)
        for category_set in word_category_sets:
            if character_category in category_set:
                c_vec.append(1)
            else:
                c_vec.append(0)
        ret_vecs.append(c_vec)
    return(ret_vecs)

def get_accuracies(softmax_predictions, gold_labels):
    assert len(softmax_predictions) == len(gold_labels), "Mismatched input and output"
    model_preds = np.argmax(softmax_predictions, axis=1)
    gold_preds = np.argmax(gold_labels, axis=1)
    confusion_mat = [[0 for i in range(5)] for j in range(5)]
    correct_counts = {}
    tot_gold_counts = {}
    tot_pred_counts = {}
    for index, model_pred in enumerate(model_preds):
        gold_pred = gold_preds[index]
        confusion_mat[model_pred][gold_pred] += 1
        tot_gold_counts[gold_pred] = tot_gold_counts.get(gold_pred, 0.0) + 1
        tot_pred_counts[model_pred] = tot_pred_counts.get(model_pred, 0.0) + 1
        if model_pred == gold_pred:
            correct_counts[model_pred] = correct_counts.get(model_pred, 0.0) + 1
    recall = {label : correct_counts.get(label, 0.0) / tot_gold_counts[label] for label in tot_gold_counts.keys()}
    precision = {label: correct_counts.get(label, 0.0) / tot_pred_counts[label] for label in tot_pred_counts.keys()}
    return(recall, precision, confusion_mat)


def get_char_composition(model, train_instances, rev_train_instances, name_prefix):
    model.add_input(name_prefix + "_forward_input", (
        train_instances.shape[1], train_instances.shape[2]))  # train_instances.shape[2] vs. vec_size?
    model.add_input(name_prefix + "_backward_input",
                    (rev_train_instances.shape[1], rev_train_instances.shape[2]))
    print("Added " + name_prefix + " inputs")

    # Using time distributed dense instead of embedding layer
    forward_embed = TimeDistributedDense(output_dim=EMBEDDING_DIM, weights=None, W_regularizer=None,
                                         b_regularizer=None, input_shape=(
            train_instances.shape[1], train_instances.shape[2]))
    backward_embed = TimeDistributedDense(output_dim=EMBEDDING_DIM, weights=None, W_regularizer=None,
                                          b_regularizer=None, input_shape=(
            rev_train_instances.shape[1], rev_train_instances.shape[2]))
    model.add_node(forward_embed, input=name_prefix + "_forward_input", name=name_prefix + "_forward_embed")
    model.add_node(backward_embed, input=name_prefix + "_backward_input", name=name_prefix + "_backward_embed")
    print("Added " + name_prefix + " embedding layers")

    # SHOULD THE TIME DISTRIBUTED LAYER BE COMMON?

    forward_LSTM = LSTM(output_dim=LSTM_OUTPUT_DIM, dropout_W=0.1, dropout_U=0.1, W_regularizer=l2(LSTM_REG_WEIGHT),
                        return_sequences=False)  # Try by pooling sequence of outputs as well
    backward_LSTM = LSTM(output_dim=LSTM_OUTPUT_DIM, dropout_W=0.1, dropout_U=0.1,
                         W_regularizer=l2(LSTM_REG_WEIGHT),
                         return_sequences=False)  # Try by pooling sequence of outputs as well

    model.add_node(forward_LSTM, input=name_prefix + "_forward_embed", name=name_prefix + "forward_lstm")
    model.add_node(backward_LSTM, input=name_prefix + "_backward_embed", name=name_prefix + "backward_lstm")

    print("Added " + name_prefix + " LSTM layers")

    return ([[name_prefix + "forward_lstm", name_prefix + "backward_lstm"], {
        name_prefix + "_forward_input" : train_instances,
        name_prefix + "_backward_input": rev_train_instances
        }])


def get_all_ipa_vocabs_and_ner_sets(file_list, epi, word_len_threshold, ignore_words=False):
    assert len(file_list) > 0, "No files specified!"
    vocab = set()
    NER_sets_dict = {}
    non_NER_set = set()
    for file in file_list:
        vocab, NER_sets_dict, non_NER_set = get_ipa_charset_and_ner_sets(vocab,
                                                                         NER_sets_dict, non_NER_set,
                                                                         file, epi,
                                                                         threshold=word_len_threshold, ignore_words=ignore_words)
    return(vocab, NER_sets_dict, non_NER_set)

def get_all_words_charset_and_ner_sets(file_list,  word_len_threshold, wlen_dicts={}, ignore_words=False):
    assert len(file_list) > 0, "No files specified"
    vocab = set()
    NER_sets_dict = {}
    non_NER_set = set()
    for file in file_list:
        vocab, NER_sets_dict, non_NER_set, wlen_dicts = get_words_charset_and_ner_sets(vocab, NER_sets_dict, non_NER_set,
                                                                                        file, ignore_words=ignore_words,
                                                                                        word_len_dict=wlen_dicts,
                                                                                        threshold=word_len_threshold)
    return(vocab, NER_sets_dict, non_NER_set, wlen_dicts)

def visualize_word_counts(wlen_dicts):
    print("Word len count stats:")
    print("Max len = %d" % (max(wlen_dicts.keys())))
    print("Min len = %d" % (min(wlen_dicts.keys())))
    med_pos = len(wlen_dicts.keys()) / 2
    sorted_counts = wlen_dicts.keys()
    sorted_counts.sort()
    median = sorted_counts[med_pos]
    print("No. of unique word counts = %d" % (len(sorted_counts)))
    print("words of Median len = %d, med*05 = %d, med*2 = %d, med*5 = %d" % (
    wlen_dicts.get(median, -1), wlen_dicts.get(int(median * 0.5), -1), wlen_dicts.get(median * 2, -1),
    wlen_dicts.get(median * 5, -1)))
    d = wlen_dicts
    X = d.keys()
    X.sort(reverse=True)
    Y = [d[x] for x in X]
    pl.bar(X, Y, align='center', width=0.5)
    pl.xticks(X, X)
    ymax = max(Y) + 1
    pl.ylim(0, ymax)
    pl.show()


def get_left_padded_numpy(vec_len, instances, max_len, reverse=False):
    for index, instance in enumerate(instances):
        zeros = [[0 for j in range(vec_len)] for i in range(max_len - len(instance))]
        if reverse:
            instances[index] = zeros + instance[::-1]
        else:
            instances[index] = zeros + instance
    return (np.array(instances))

def create_phono_mats(phono_src_char_vocab, word_category_sets, NER_sets_dict, non_NER_set, epi, use_caps=True):
    print("Creating phono mats")
    phono_train_instances = []
    phono_test_instances = []
    phono_train_labels = []
    phono_test_labels = []

    phono_label_index_dict = {}

    phono_max_len = 0

    n_classes = len(NER_sets_dict.keys()) + 1

    for tag_index, ner_tag in enumerate(NER_sets_dict.keys()):
        phono_label_index_dict[ner_tag] = tag_index
        ner_words = NER_sets_dict[ner_tag]
        tag_instances = []
        tag_labels = []
        for word in ner_words:
            tag_instances.append(get_instance_from_word(word, epi, phono_src_char_vocab, word_category_sets, use_caps=use_caps))
            zero_label_vect = [0 for i in range(n_classes)]
            zero_label_vect[tag_index] = 1
            assert tag_index < len(zero_label_vect) - 1, "Non ner class label given wrongly"
            tag_labels.append(zero_label_vect)
            if len(word) > phono_max_len:
                phono_max_len = len(word)
        split_point = int(len(tag_instances) * (1 - TEST_SPLIT))
        phono_train_instances.extend(tag_instances[:split_point])
        phono_train_labels.extend(tag_labels[:split_point])
        phono_test_instances.extend(tag_instances[split_point:])
        phono_test_labels.extend(tag_labels[split_point:])

    non_ner_instances = []
    non_ner_labels = []
    for word in non_NER_set:
        non_ner_instances.append(get_instance_from_word(word, epi, phono_src_char_vocab, word_category_sets, use_caps=use_caps))
        zero_label_vect = [0 for i in range(n_classes)]
        zero_label_vect[-1] = 1
        non_ner_labels.append(zero_label_vect)
        if len(word) > phono_max_len:
            phono_max_len = len(word)

    print("Phono max len is %d" % (phono_max_len))

    split_point = int(len(non_ner_instances) * (1 - TEST_SPLIT))
    phono_train_instances.extend(non_ner_instances[:split_point])
    phono_train_labels.extend(non_ner_labels[:split_point])
    phono_test_instances.extend(non_ner_instances[split_point:])
    phono_test_labels.extend(non_ner_labels[split_point:])

    phono_rev_train_instances = list(phono_train_instances)
    phono_rev_test_instances = list(phono_test_instances)

    vec_len = len(phono_train_instances[0][0])

    phono_train_instances = get_left_padded_numpy(vec_len=vec_len, instances=phono_train_instances, max_len=phono_max_len)
    phono_rev_train_instances = get_left_padded_numpy(vec_len=vec_len, instances=phono_rev_train_instances, max_len=phono_max_len, reverse=True)
    phono_test_instances = get_left_padded_numpy(vec_len=vec_len, instances=phono_test_instances, max_len=phono_max_len)
    phono_rev_test_instances = get_left_padded_numpy(vec_len=vec_len, instances=phono_rev_test_instances, max_len=phono_max_len, reverse=True)

    phono_train_labels = np.array(phono_train_labels)
    phono_test_labels = np.array(phono_test_labels)

    print("Phono label to index dict is")
    print(phono_label_index_dict)

    print("Train input shapes")
    print(phono_train_instances.shape)
    print(phono_rev_train_instances.shape)

    print("Test input shapes")
    print(phono_test_instances.shape)
    print(phono_rev_test_instances.shape)
    return([phono_train_instances, phono_rev_train_instances, phono_train_labels],
           [phono_test_instances, phono_rev_test_instances, phono_test_labels],
           phono_label_index_dict)

def create_ortho_mats(ortho_src_char_vocab, NER_sets_dict, non_NER_set):
    print("Getting mats for ortho")
    n_classes = len(NER_sets_dict) + 1
    ortho_max_len = 0
    ortho_train_instances = []
    ortho_test_instances = []
    ortho_train_labels = []
    ortho_test_labels = []
    ortho_label_index_dict = {}
    for tag_index, ner_tag in enumerate(NER_sets_dict.keys()):
        ortho_label_index_dict[ner_tag] = tag_index
        ner_words = NER_sets_dict[ner_tag]
        ortho_tag_instances = []
        ortho_tag_labels = []
        for word in ner_words:
            ortho_tag_instances.append(
                [get_one_hot_char(len(ortho_src_char_vocab.keys()), ortho_src_char_vocab[char]) for char in word])
            zero_label_vect = [0 for i in range(n_classes)]
            zero_label_vect[tag_index] = 1
            assert tag_index < len(zero_label_vect) - 1, "Non ner class label given wrongly"
            ortho_tag_labels.append(zero_label_vect)
            if len(word) > ortho_max_len:
                ortho_max_len = len(word)
        split_point = int(len(ortho_tag_instances) * (1 - TEST_SPLIT))
        ortho_train_instances.extend(ortho_tag_instances[:split_point])
        ortho_train_labels.extend(ortho_tag_labels[:split_point])
        ortho_test_instances.extend(ortho_tag_instances[split_point:])
        ortho_test_labels.extend(ortho_tag_labels[split_point:])

    ortho_non_ner_instances = []
    ortho_non_ner_labels = []
    for word in non_NER_set:
        ortho_non_ner_instances.append(
            [get_one_hot_char(len(ortho_src_char_vocab.keys()), ortho_src_char_vocab[char]) for char in word])
        zero_label_vect = [0 for i in range(n_classes)]
        zero_label_vect[-1] = 1
        ortho_non_ner_labels.append(zero_label_vect)
        if len(word) > ortho_max_len:
            ortho_max_len = len(word)

    print("Ortho max len is %d" % (ortho_max_len))

    split_point = int(len(ortho_non_ner_instances) * (1 - TEST_SPLIT))
    ortho_train_instances.extend(ortho_non_ner_instances[:split_point])
    ortho_train_labels.extend(ortho_non_ner_labels[:split_point])
    ortho_test_instances.extend(ortho_non_ner_instances[split_point:])
    ortho_test_labels.extend(ortho_non_ner_labels[split_point:])

    ortho_rev_train_instances = list(ortho_train_instances)
    ortho_rev_test_instances = list(ortho_test_instances)
    print(len(ortho_test_instances))
    print(len(ortho_rev_test_instances))

    vec_len = len(ortho_train_instances[0][0])

    ortho_train_instances = get_left_padded_numpy(vec_len, ortho_train_instances,ortho_max_len)
    ortho_rev_train_instances = get_left_padded_numpy(vec_len, ortho_rev_train_instances, ortho_max_len, reverse=True)
    ortho_test_instances = get_left_padded_numpy(vec_len, ortho_test_instances, ortho_max_len)
    ortho_rev_test_instances = get_left_padded_numpy(vec_len, ortho_rev_test_instances, ortho_max_len, reverse=True)
    ortho_train_labels = np.array(ortho_train_labels)
    ortho_test_labels = np.array(ortho_test_labels)

    print("Label to index dict for ortho is:")
    print(ortho_label_index_dict)

    print("Ortho Train input shapes")
    print(ortho_train_instances.shape)
    print(ortho_rev_train_instances.shape)

    print("Ortho Test input shapes")
    print(ortho_test_instances.shape)
    print(ortho_rev_test_instances.shape)
    return([ortho_train_instances, ortho_rev_train_instances, ortho_train_labels], [ortho_test_instances, ortho_rev_test_instances, ortho_test_labels], ortho_label_index_dict)


def eval(fixed_char_encoder, phono_instances, phono_rev_instances, ortho_instances, ortho_rev_instances, phono_labels, ortho_labels, wvec_instances, wvec_labels):
    print("Sanity checks")
    if USE_PHONO and USE_ORTHO:
        assert False not in np.equal(ortho_labels, phono_labels), "Test sets not equal"
        if USE_WORD_VECS:
            assert False not in np.equal(ortho_labels, wvec_labels), "Test sets not equal"
        print("Sanity check passed")
    elif USE_PHONO:
        if USE_WORD_VECS:
            assert False not in np.equal(wvec_labels, phono_labels), "Test sets not equal"
        print("Sanity check passed")
    elif USE_ORTHO:
        if USE_WORD_VECS:
            assert False not in np.equal(ortho_labels, wvec_labels), "Test sets not equal"
        print("Sanity check passed")
    predict_inputs = {}
    if USE_PHONO:
        predict_inputs["phono_forward_input"] = phono_instances
        predict_inputs["phono_backward_input"] = phono_rev_instances
    if USE_ORTHO:
        predict_inputs["ortho_forward_input"] = ortho_instances
        predict_inputs["ortho_backward_input"] = ortho_rev_instances
    if USE_WORD_VECS:
        predict_inputs["wvec_input"] = wvec_instances

    predictions = fixed_char_encoder.predict(
        predict_inputs)  # {'output':...}
    # name_prefix + "_forward_input"
    print("Obtained predictions")
    # print(predictions)

    ner_outputs = predictions['ner_output']

    print("Obtained ner_outputs")
    # print(ner_outputs)
    print("Shape of ner_outputs:", ner_outputs.shape)

    print("Outputs shape: ", ner_outputs.shape)
    if USE_PHONO:
        print("Labels shape: ", phono_labels.shape)
        print("Instances shape: ", phono_instances.shape, phono_rev_instances.shape)
        recall, precision, test_conf_mat = get_accuracies(ner_outputs, phono_labels)
    elif USE_ORTHO:
        print("Labels shape: ", ortho_labels.shape)
        print("Instances shape: ", ortho_instances.shape, ortho_rev_instances.shape)
        recall, precision, test_conf_mat = get_accuracies(ner_outputs, ortho_labels)

    print("Stats")
    print("Recall")
    print(recall)
    print("Precision")
    print(precision)
    print(test_conf_mat)

def read_cca_vecs(cca_vec_file):
    vec_dict = {}
    with open(cca_vec_file, 'r') as vec_file:
        for line in vec_file:
            line = line.strip(" \t\r\n")
            if len(line) > 0:
                split_line = line.split()
                # assert len(split_line[0].split(":")) == 2, "Too many splits in vec word"
                # if len(split_line[0].split(":")) > 2:
                #     print(split_line[0])
                #     word = "".join(split_line[0].split(":")[1:])
                #     print(word)
                #     raw_input("We got a problem")
                # else:
                #     word = split_line[0].split(":")[1]
                word = split_line[0].split(":", 1)[-1].strip(" \t\r\n").decode('utf8')
                # if len(split_line[0].split(":")) > 2:
                #     print(split_line[0])
                #     print(word)
                #     raw_input("Problem case")
                vec = np.array([float(elt) for elt in split_line[1:]])
                vec_dict[word] = vec
    return(vec_dict)

def create_wvec_mats(NER_sets_dict, non_NER_set, wvec_dict, test_split):
    wvec_train_instances = []
    wvec_train_labels = []
    wvec_test_instances = []
    wvec_test_labels = []
    wvec_label_index_dict = {}
    n_classes = len(NER_sets_dict) + 1
    wvec_len = len(wvec_dict[wvec_dict.keys()[0]])
    ner_skip_count = 0
    for tag_index, ner_tag in enumerate(NER_sets_dict.keys()):
        wvec_label_index_dict[ner_tag] = tag_index
        ner_words = NER_sets_dict[ner_tag]
        wvec_tag_instances = []
        wvec_tag_labels = []
        for word in ner_words:
            try:
                wvec_tag_instances.append(wvec_dict[word])
            except:
                # print("%s NER's word vector not found. Using 0 vector instead"%(word))
                ner_skip_count += 1
                wvec_tag_instances.append(np.zeros(wvec_len, dtype=np.float))
            zero_label_vect = [0 for i in range(n_classes)]
            zero_label_vect[tag_index] = 1
            assert tag_index < len(zero_label_vect) - 1, "Non ner class label given wrongly"
            wvec_tag_labels.append(zero_label_vect)
        split_point = int(len(wvec_tag_instances) * (1 - test_split))
        wvec_train_instances.extend(wvec_tag_instances[:split_point])
        wvec_train_labels.extend(wvec_tag_labels[:split_point])
        wvec_test_instances.extend(wvec_tag_instances[split_point:])
        wvec_test_labels.extend(wvec_tag_labels[split_point:])

    non_ner_skip_count = 0
    wvec_non_ner_instances = []
    wvec_non_ner_labels = []
    for word in non_NER_set:
        try:
            wvec_non_ner_instances.append(wvec_dict[word])
        except:
            # print("%s non-NER's word vector not found. Using 0 vector instead" % (word))
            non_ner_skip_count += 1
            wvec_non_ner_instances.append(np.zeros(wvec_len, dtype=np.float))
        zero_label_vect = [0 for i in range(n_classes)]
        zero_label_vect[-1] = 1
        wvec_non_ner_labels.append(zero_label_vect)

    split_point = int(len(wvec_non_ner_instances) * (1 - test_split))
    wvec_train_instances.extend(wvec_non_ner_instances[:split_point])
    wvec_train_labels.extend(wvec_non_ner_labels[:split_point])
    wvec_test_instances.extend(wvec_non_ner_instances[split_point:])
    wvec_test_labels.extend(wvec_non_ner_labels[split_point:])

    print("Word vector label to index dict")
    print(wvec_label_index_dict)
    print("%d NER's word vectors not found. %d non-NER's word vectors not found. They have been zeroed out"%(ner_skip_count, non_ner_skip_count))

    return([np.array(wvec_train_instances), np.array(wvec_train_labels)], [np.array(wvec_test_instances), np.array(wvec_test_labels)])


# CONFIRM THE WORD ORDERS ARE THE SAME FOR ORTHO, PHONO AND WVEC, perhaps by returning the words in order of processing from each extraction step

def train_model():
    assert USE_PHONO or USE_ORTHO, "Not using any character representations!"

    if USE_WORD_VECS:
        src_wvecs = read_cca_vecs(SRC_WVECS)
        trg_wvecs = read_cca_vecs(TRG_WVECS)
        print(src_wvecs["Dinarʼda".decode('utf8')])

    if USE_PHONO:
        print("Extracting vocabs for Phono")
        assert USE_SRC or USE_TRG, "Target or source IPA vocab choice not specified"
        if USE_SRC and not USE_TRG:
            epi_targ = epi = epitran.Epitran(SRC_LANG)
        elif USE_TRG and not USE_SRC:
            epi = epi_targ = epitran.Epitran(TRG_LANG)
        elif USE_SRC and USE_TRG:
            epi = epitran.Epitran(SRC_LANG)
            epi_targ = epitran.Epitran(TRG_LANG)

        phono_src_char_vocab, NER_sets_dict, non_NER_set = get_all_ipa_vocabs_and_ner_sets([TRAIN_CONLL,
                                                                                           DEV_CONLL, TEST_CONLL], epi, WORD_LEN_THRESHOLD)
        phono_targ_char_vocab, targ_NER_sets_dict, targ_non_NER_set = get_all_ipa_vocabs_and_ner_sets(
            [TARGET_TRAIN_CONLL, TARGET_DEV_CONLL, TARGET_TEST_CONLL], epi_targ, WORD_LEN_THRESHOLD)

        if phono_targ_char_vocab == phono_src_char_vocab:
            print("Equal vocabs case")
        if (phono_targ_char_vocab.intersection(phono_src_char_vocab) == phono_targ_char_vocab):
            print("Targ vocab is smaller")
        elif (phono_src_char_vocab.intersection(phono_targ_char_vocab) == phono_src_char_vocab):
            print("Source vocab is smaller")
        elif len(phono_src_char_vocab.intersection(phono_targ_char_vocab)) > 0:
            print("Overlap of %d words, source vab size is %d, targ vocab size is %d" % (
            len(phono_src_char_vocab.intersection(phono_targ_char_vocab)), len(phono_src_char_vocab),
            len(phono_targ_char_vocab)))
        else:
            print("No overlap")

        phono_src_char_vocab = phono_src_char_vocab.union(phono_targ_char_vocab)

        phono_src_char_vocab = {char: char_index for char_index, char in enumerate(phono_src_char_vocab)}
        phono_src_char_vocab[-1] = len(
            phono_src_char_vocab.keys())  # ----> confirm!! should also handle puncts properly
        print("Final phono char vocab has size: %d" % (len(phono_src_char_vocab.keys())))
        pickle.dump(phono_src_char_vocab, open(MODEL_NAME_PREFIX + "_phono_char_vocab.vocab", 'wb'))
        print("Dumped phono vocab")



    if USE_ORTHO:
        print("Extracting char vocab for ortho")
        src_ortho_char_vocab, _asrc, _bsrc, wlen_dicts = get_all_words_charset_and_ner_sets([TRAIN_CONLL, DEV_CONLL,
                                                                                       TEST_CONLL], WORD_LEN_THRESHOLD,
                                                                                       ignore_words=USE_PHONO)

        trg_ortho_char_vocab, _atrg, _btrg, wlen_dicts = get_all_words_charset_and_ner_sets([TARGET_TRAIN_CONLL,
                                                                                       TARGET_DEV_CONLL,
                                                                                       TARGET_TEST_CONLL],
                                                                                       WORD_LEN_THRESHOLD,
                                                                                       wlen_dicts=wlen_dicts,
                                                                                       ignore_words=USE_PHONO)

        if USE_PHONO:
            _asrc = _bsrc = _atrg = _btrg = None
        else:
            NER_sets_dict = _asrc
            non_NER_set = _bsrc
            targ_NER_sets_dict = _atrg
            targ_non_NER_set = _btrg

        src_ortho_char_vocab = src_ortho_char_vocab.union(trg_ortho_char_vocab)

        src_ortho_char_vocab = {char: char_index for char_index, char in enumerate(src_ortho_char_vocab)}

        src_ortho_char_vocab = {char: char_index for char_index, char in enumerate(src_ortho_char_vocab)}
        src_ortho_char_vocab[-1] = len(src_ortho_char_vocab.keys())
        print("Final ortho char vocab has size: %d" % (len(src_ortho_char_vocab.keys())))
        pickle.dump(src_ortho_char_vocab, open(MODEL_NAME_PREFIX + "_ortho_char_vocab.vocab", 'wb'))
        print("Dumped ortho vocab")

    print("Vocab extraction done")

    if USE_WORD_VECS:
        print("Obtaining word vector train/test matrices")
        print(src_wvecs["Dinarʼda".decode('utf8')])
        [wvec_train_instances, wvec_train_labels], [wvec_test_instances, wvec_test_labels] = create_wvec_mats(
            NER_sets_dict, non_NER_set, src_wvecs, TEST_SPLIT)
        print("Word vector matrices obtained")
        print(wvec_train_instances.shape, wvec_train_labels.shape)
        print(wvec_test_instances.shape, wvec_test_labels.shape)
    else:
        wvec_train_instances = None
        wvec_train_labels = None
        wvec_test_instances = None
        wvec_test_labels = None

    char_codes = set(["Lu", "Ll", "Lt", "Lm", "Lo"])
    number_codes = set(["Nd", "Nl", "No"])
    punc_codes = set(["Pc", "Pc", "Ps", "Pe", "Pi", "Pf", "Po"])
    sym_codes = set(["Sm", "Sc", "Sk", "So"])
    sep_codes = set(["Zs", "Zl", "Zp"])
    mark_codes = set(["Mn", "Mc", "Me"])
    other_codes = set(["Cc", "Cs", "Cf", "Co", "Cn"])
    if USE_CATS:
        word_category_sets = [char_codes, number_codes, punc_codes, sym_codes, sep_codes, mark_codes, other_codes]
    else:
        word_category_sets = [] # No word categories used

    n_classes = len(NER_sets_dict.keys()) + 1

    if USE_PHONO:
        [phono_train_instances,
         phono_rev_train_instances,
         phono_train_labels], \
        [phono_test_instances,
         phono_rev_test_instances,
         phono_test_labels], \
        phono_label_index_dict =\
            create_phono_mats(phono_src_char_vocab, word_category_sets,NER_sets_dict, non_NER_set, epi, use_caps=USE_CAPS)
    else:
        phono_train_instances = None
        phono_rev_train_instances = None
        phono_train_labels = None
        phono_test_instances = None
        phono_rev_test_instances = None
        phono_test_labels = None
        phono_label_index_dict = None

    # Compiling ortho feats

    if USE_ORTHO:
        [ortho_train_instances,
         ortho_rev_train_instances,
         ortho_train_labels], \
        [ortho_test_instances,
         ortho_rev_test_instances,
         ortho_test_labels], \
        ortho_label_index_dict = \
            create_ortho_mats(src_ortho_char_vocab, NER_sets_dict, non_NER_set)
    else:
        ortho_train_instances = None
        ortho_rev_train_instances = None
        ortho_train_labels = None
        ortho_test_instances = None
        ortho_rev_test_instances = None
        ortho_test_labels = None
        ortho_label_index_dict = None

    # CONFIRM IF INPUTS ARE CORRECT!

    """
    Try using lstm output at each sequence and convolving over it.
    Also try grouping datasets based on length of sequence for training.
    """

    if USE_PHONO and USE_ORTHO:
        print("Sanity checks")
        assert False not in np.equal(ortho_train_labels, phono_train_labels), "Train sets are not equal"
        assert False not in np.equal(ortho_test_labels, phono_test_labels), "Test sets not equal"
        if USE_WORD_VECS:
            assert False not in np.equal(ortho_train_labels, wvec_train_labels), "Wvec train set doesn't match"
            assert False not in np.equal(ortho_test_labels, wvec_test_labels), "Wvec test set doesn't match"
        print("Sanity check passed")
    elif USE_PHONO:
        ortho_test_labels = phono_test_labels
        ortho_train_labels = phono_train_labels
        if USE_WORD_VECS:
            assert False not in np.equal(phono_train_labels, wvec_train_labels), "Wvec train set doesn't match"
            assert False not in np.equal(phono_test_labels, wvec_test_labels), "Wvec test set doesn't match"
    elif USE_ORTHO:
        phono_train_labels = ortho_train_labels
        phono_test_labels = ortho_test_labels
        if USE_WORD_VECS:
            assert False not in np.equal(ortho_train_labels, wvec_train_labels), "Wvec train set doesn't match"
            assert False not in np.equal(ortho_test_labels, wvec_test_labels), "Wvec test set doesn't match"

    print("Creating model")
    fixed_char_encoder = Graph()

    merge_reps = []
    inputs = {}

    if USE_PHONO:
        [[phono_forward_lstm, phono_backward_lstm], phono_inputs] = get_char_composition(fixed_char_encoder,
                                                                                         phono_train_instances,
                                                                                         phono_rev_train_instances,
                                                                                         "phono")
        merge_reps += [phono_forward_lstm, phono_backward_lstm]
        inputs.update(phono_inputs)
    if USE_ORTHO:
        [[ortho_forward_lstm, ortho_backward_lstm], ortho_inputs]= get_char_composition(fixed_char_encoder,
                                                                                        ortho_train_instances,
                                                                                        ortho_rev_train_instances,
                                                                                        "ortho")
        merge_reps += [ortho_forward_lstm, ortho_backward_lstm]
        inputs.update(ortho_inputs)
    if USE_WORD_VECS:
        print("Adding the word vector input")
        fixed_char_encoder.add_input("wvec_input", (wvec_train_instances.shape[1],))
        inputs["wvec_input"] = wvec_train_instances
        merge_reps += ["wvec_input"]
        print("Added the word vector input")

    pre_predict_layer = Dense(output_dim=PRE_PREDICT_DIM, activation="tanh", W_regularizer=l2(PRE_PRED_REG))

    fixed_char_encoder.add_node(pre_predict_layer, name="pre_predict", inputs=merge_reps, merge_mode="concat")

    print("Added pre_predict layer")

    predict_layer = Dense(output_dim=n_classes, activation="softmax", W_regularizer=l2(PRED_REG))
    fixed_char_encoder.add_node(predict_layer, name="predict_output", input="pre_predict")

    print("Added predict layer")

    fixed_char_encoder.add_output(name="ner_output",input="predict_output")

    print("Added output. Now compiling model")

    fixed_char_encoder.compile(optimizer=RMSprop(lr=0.001, rho=0.5, epsilon=1e-06), loss={"ner_output":'categorical_crossentropy'})

    print("Compiled model. Now fitting data")

    fit_inputs = {}
    fit_inputs.update(inputs)
    fit_inputs["ner_output"] = phono_train_labels

    history = fixed_char_encoder.fit(data=fit_inputs, batch_size=N_BATCH, nb_epoch=N_EPOCH, validation_split=0.1)

    print("Evaluating on test")
    eval(fixed_char_encoder, phono_test_instances, phono_rev_test_instances, ortho_test_instances,
         ortho_rev_test_instances, phono_test_labels, ortho_test_labels, wvec_test_instances, wvec_test_labels)

    print("Saving model architecture")

    with open(MODEL_NAME_PREFIX + str(N_EPOCH) + "_" + str(N_BATCH) +  ".arch", 'wb') as model_file:
        model_file.write(fixed_char_encoder.to_json())

    print("Saving model weights")

    fixed_char_encoder.save_weights(MODEL_NAME_PREFIX + str(N_EPOCH) + "_" + str(N_BATCH) + ".weights.h5", overwrite=True)

    # predictions = fixed_char_encoder.predict({"forward_input":phono_train_instances, "backward_input":phono_rev_train_instances}) # {'output':...}
    print("Evaluating on train")
    eval(fixed_char_encoder, phono_train_instances, phono_rev_train_instances, ortho_train_instances,
         ortho_rev_train_instances, phono_train_labels, ortho_train_labels, wvec_train_instances, wvec_train_labels)


    if USE_PHONO:
        print phono_label_index_dict
    elif USE_ORTHO:
        print(ortho_label_index_dict)

    raw_input("Transfer is up next. Press enter to continue")
    # raw_input("Finished training. Press enter to transfer model")

    print("De allocating source language mats to regain memory")
    del phono_train_labels
    del phono_train_instances
    del phono_test_labels
    del phono_test_instances
    del phono_rev_train_instances
    del phono_rev_test_instances

    del ortho_train_labels
    del ortho_train_instances
    del ortho_test_instances
    del ortho_test_labels
    del ortho_rev_train_instances
    del ortho_rev_test_instances
    del wvec_train_instances
    del wvec_train_labels
    del wvec_test_instances
    del wvec_test_labels

    if USE_PHONO:
        [phono_train_instances,
         phono_rev_train_instances,
         phono_train_labels], \
        [phono_test_instances,
         phono_rev_test_instances,
         phono_test_labels], \
        phono_label_index_dict = \
            create_phono_mats(phono_src_char_vocab, word_category_sets, targ_NER_sets_dict, targ_non_NER_set, epi_targ, use_caps=USE_CAPS)
    else:
        phono_train_instances = None
        phono_rev_train_instances = None
        phono_train_labels = None
        phono_test_instances = None
        phono_rev_test_instances = None
        phono_test_labels = None
        phono_label_index_dict = None


    # Compiling ortho feats

    if USE_ORTHO:
        [ortho_train_instances,
         ortho_rev_train_instances,
         ortho_train_labels], \
        [ortho_test_instances,
         ortho_rev_test_instances,
         ortho_test_labels], \
        ortho_label_index_dict = \
            create_ortho_mats(src_ortho_char_vocab, targ_NER_sets_dict, targ_non_NER_set)
    else:
        ortho_train_instances = None
        ortho_rev_train_instances = None
        ortho_train_labels = None
        ortho_test_instances = None
        ortho_rev_test_instances = None
        ortho_test_labels = None
        ortho_label_index_dict = None

    if USE_WORD_VECS:
        [wvec_train_instances, wvec_train_labels], [wvec_test_instances, wvec_test_labels] = create_wvec_mats(
            targ_NER_sets_dict, targ_non_NER_set, trg_wvecs, TEST_SPLIT)
    else:
        wvec_train_instances = None
        wvec_train_labels = None
        wvec_test_instances = None
        wvec_test_labels = None

    print("Evaluating on test")
    eval(fixed_char_encoder, phono_test_instances, phono_rev_test_instances, ortho_test_instances,
         ortho_rev_test_instances, phono_test_labels, ortho_test_labels, wvec_test_instances, wvec_test_labels)

    target_fit_inputs = {}
    if USE_PHONO:
        target_fit_inputs["phono_forward_input"] = phono_train_instances
        target_fit_inputs["phono_backward_input"] = phono_rev_train_instances
    if USE_ORTHO:
        target_fit_inputs["ortho_forward_input"] = ortho_train_instances
        target_fit_inputs["ortho_backward_input"] = ortho_rev_train_instances
    if USE_WORD_VECS:
        target_fit_inputs["wvec_input"] = wvec_train_instances

    if USE_PHONO:
        target_fit_inputs["ner_output"] = phono_train_labels
    elif USE_ORTHO:
        target_fit_inputs["ner_output"] = ortho_train_labels

    history = fixed_char_encoder.fit(data=target_fit_inputs, batch_size=N_BATCH, nb_epoch=N_EPOCH, validation_split=0.1)

    print("Testing after fitting to target language data")

    print("Evaluating on test")
    eval(fixed_char_encoder, phono_test_instances, phono_rev_test_instances, ortho_test_instances,
         ortho_rev_test_instances, phono_test_labels, ortho_test_labels, wvec_test_instances, wvec_test_labels)

    print("Saving model architecture")

    with open(MODEL_NAME_PREFIX + str(N_EPOCH) + "_" + str(N_BATCH) + "after_target_fit" + ".arch", 'wb') as model_file:
        model_file.write(fixed_char_encoder.to_json())

    print("Saving model weights")

    fixed_char_encoder.save_weights(MODEL_NAME_PREFIX + str(N_EPOCH) + "_" + str(N_BATCH) + "after_target_fit" + ".weights.h5",
                                    overwrite=True)

train_model()