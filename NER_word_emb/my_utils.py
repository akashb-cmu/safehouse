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

sys.path.append("./epitran/epitran/bin/")
sys.path.append("./epitran/epitran/")
import word2pfvectors
import _epitran as epitran



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

def get_instance_from_word(word, epi, char_vocab, word_category_sets):
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
    tot_counts = {}
    for index, model_pred in enumerate(model_preds):
        gold_pred = gold_preds[index]
        confusion_mat[model_pred][gold_pred] += 1
        tot_counts[gold_pred] = tot_counts.get(gold_pred, 0.0) + 1
        if model_pred == gold_pred:
            correct_counts[model_pred] = correct_counts.get(model_pred, 0.0) + 1
    accuracies = {label : correct_counts.get(label, 0.0) / tot_counts[label] for label in tot_counts.keys()}
    return(accuracies, confusion_mat)


def get_char_composition(model, train_instances, rev_train_instances, name_prefix, EMBEDDING_DIM, LSTM_OUTPUT_DIM, LSTM_REG_WEIGHT):
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

def get_all_words_charset_and_ner_sets(file_list,  word_len_threshold=sys.maxint, wlen_dicts={}, ignore_words=False):
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

def create_phono_mats(phono_src_char_vocab, word_category_sets, NER_sets_dict, non_NER_set, epi, TEST_SPLIT):
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
            tag_instances.append(get_instance_from_word(word, epi, phono_src_char_vocab, word_category_sets))
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
        non_ner_instances.append(get_instance_from_word(word, epi, phono_src_char_vocab, word_category_sets))
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

    print("Train input shapes")
    print(phono_train_instances.shape)
    print(phono_rev_train_instances.shape)

    print("Test input shapes")
    print(phono_test_instances.shape)
    print(phono_rev_test_instances.shape)
    return([phono_train_instances, phono_rev_train_instances, phono_train_labels],
           [phono_test_instances, phono_rev_test_instances, phono_test_labels],
           phono_label_index_dict)

def create_ortho_mats(ortho_src_char_vocab, NER_sets_dict, non_NER_set, TEST_SPLIT):
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

    print("Ortho Train input shapes")
    print(ortho_train_instances.shape)
    print(ortho_rev_train_instances.shape)

    print("Ortho Test input shapes")
    print(ortho_test_instances.shape)
    print(ortho_rev_test_instances.shape)
    return([ortho_train_instances, ortho_rev_train_instances, ortho_train_labels], [ortho_test_instances, ortho_rev_test_instances, ortho_test_labels], ortho_label_index_dict)


def eval(fixed_char_encoder, phono_instances, phono_rev_instances, ortho_instances, ortho_rev_instances, phono_labels, ortho_labels, USE_PHONO, USE_ORTHO):
    if USE_PHONO and USE_ORTHO:
        print("Sanity checks")
        assert False not in np.equal(ortho_labels, phono_labels), "Test sets not equal"
        print("Sanity check passed")

    predict_inputs = {}
    if USE_PHONO:
        predict_inputs["phono_forward_input"] = phono_instances
        predict_inputs["phono_backward_input"] = phono_rev_instances
    if USE_ORTHO:
        predict_inputs["ortho_forward_input"] = ortho_instances
        predict_inputs["ortho_backward_input"] = ortho_rev_instances

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
        print("Test labels shape: ", phono_labels.shape)
        print("Test instances shape: ", phono_instances.shape, phono_rev_instances.shape)
        accuracies, test_conf_mat = get_accuracies(ner_outputs, phono_labels)
    elif USE_ORTHO:
        print("Test labels shape: ", ortho_labels.shape)
        print("Test instances shape: ", ortho_instances.shape, ortho_rev_instances.shape)
        accuracies, test_conf_mat = get_accuracies(ner_outputs, ortho_labels)


    print("Test stats")

    print(accuracies)
    print(test_conf_mat)