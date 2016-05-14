# -*- coding: utf-8 -*-
from keras.models import Sequential, Graph
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.regularizers import l1,l2
from keras.optimizers import RMSprop
import numpy as np
import pickle
import sys

sys.path.append("./epitran/epitran/bin/")
sys.path.append("./epitran/epitran/")
import word2pfvectors
import _epitran as epitran

epi = epitran.Epitran("tur-Latn")
# word = 'Özturanʼın'.decode("utf-8")
# word = "Özturan.12ın".decode("utf-8")
# ret_val = epi.word_to_pfvector(word)
# raw_input("Enter to end")

TRAIN_CONLL = "./conll_data/turkish_conll_train.conll"
TEST_CONLL = "./conll_data/turkish_conll_test.conll"
DEV_CONLL = "./conll_data/turkish_conll_dev.conll"

TARGET_TRAIN_CONLL = "./conll_data/uzbek_conll_train.conll"
TARGET_TEST_CONLL = "./conll_data/uzbek_conll_test.conll"
TARGET_DEV_CONLL = "./conll_data/uzbek_conll_dev.conll"

MODEL_NAME_PREFIX = "Turkish_model_iter_100"


TEST_SPLIT = 0.1

char_vocab = set()
# char_vocab.update()
NER_sets_dict = {}
non_NER_set = set()

def get_words_charset_and_ner_sets(char_vocab, ner_sets_dict, non_ner_set, file):
    with open(file, 'r') as ip_file:
        for line in ip_file:
            line = line.strip(' \t\r\n')
            if len(line) == 0:
                continue
            word, tag = line.split("\t")
            word = word.decode('utf8')
            if 'B-' in tag or 'I-' in tag:
                ner_sets_dict[tag[2:]] = set([word]).union(ner_sets_dict.get(tag[2:], set()))
                if word in non_ner_set:
                    non_ner_set.remove(word)
            elif not any(word in ner_set for ner_set in ner_sets_dict.values()) : # MAKE NERs and non-NERs clearly separable
                non_ner_set.add(word)
            char_vocab.update(list(word))
    return(char_vocab, ner_sets_dict, non_ner_set)

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

def get_ipa_charset_and_ner_sets(ipa_vocab, ner_sets_dict, non_ner_set, file, epi):
    with open(file, 'r') as ip_file:
        for line in ip_file:
            line = line.strip(' \t\r\n')
            if len(line) == 0:
                continue
            word, tag = line.split("\t")
            w_len = len(word)
            word = word.decode('utf8')
            w_len = len(word)
            if 'B-' in tag or 'I-' in tag:
                ner_sets_dict[tag[2:]] = set([word]).union(ner_sets_dict.get(tag[2:], set()))
                if word in non_ner_set:
                    non_ner_set.remove(word)
            elif not any(word in ner_set for ner_set in ner_sets_dict.values()) : # MAKE NERs and non-NERs clearly separable
                non_ner_set.add(word)
            ipa_vocab.update(get_ipa_ids(word, epi))
            # ipa_vocab.update(list(word))
    return(ipa_vocab, ner_sets_dict, non_ner_set)

targ_char_vocab = set()
targ_NER_sets_dict = {}
targ_non_NER_set = set()
# char_vocab, NER_sets_dict, non_NER_set = get_words_charset_and_ner_sets(char_vocab, NER_sets_dict, non_NER_set, TRAIN_CONLL)
# char_vocab, NER_sets_dict, non_NER_set = get_words_charset_and_ner_sets(char_vocab, NER_sets_dict, non_NER_set, DEV_CONLL)
# char_vocab, NER_sets_dict, non_NER_set = get_words_charset_and_ner_sets(char_vocab, NER_sets_dict, non_NER_set, TEST_CONLL)

char_vocab, NER_sets_dict, non_NER_set = get_ipa_charset_and_ner_sets(char_vocab, NER_sets_dict, non_NER_set, TRAIN_CONLL, epi)
char_vocab, NER_sets_dict, non_NER_set = get_ipa_charset_and_ner_sets(char_vocab, NER_sets_dict, non_NER_set, DEV_CONLL, epi)
char_vocab, NER_sets_dict, non_NER_set = get_ipa_charset_and_ner_sets(char_vocab, NER_sets_dict, non_NER_set, TEST_CONLL, epi)

epi_targ = epitran.Epitran("uzb-Latn")
targ_char_vocab, targ_NER_sets_dict, targ_non_NER_set = get_ipa_charset_and_ner_sets(targ_char_vocab, targ_NER_sets_dict, targ_non_NER_set, TARGET_TRAIN_CONLL, epi_targ)
targ_char_vocab, targ_NER_sets_dict, targ_non_NER_set = get_ipa_charset_and_ner_sets(targ_char_vocab, targ_NER_sets_dict, targ_non_NER_set, TARGET_DEV_CONLL, epi_targ)
targ_char_vocab, targ_NER_sets_dict, targ_non_NER_set = get_ipa_charset_and_ner_sets(targ_char_vocab, targ_NER_sets_dict, targ_non_NER_set, TARGET_TEST_CONLL, epi_targ)

if targ_char_vocab == char_vocab:
    print("Equal vocabs case")
if (targ_char_vocab.intersection(char_vocab) == targ_char_vocab):
    print("Targ vocab is smaller")
elif  (char_vocab.intersection(targ_char_vocab) == char_vocab):
    print("Source vocab is smaller")
elif len(char_vocab.intersection(targ_char_vocab)) > 0:
    print("Overlap of %d words, source vab size is %d, targ vocab size is %d"%(len(char_vocab.intersection(targ_char_vocab)), len(char_vocab), len(targ_char_vocab)))
else:
    print("No overlap")

char_vocab = char_vocab.union(targ_char_vocab)

char_vocab = {char : char_index for char_index, char in enumerate(char_vocab)}
char_vocab[-1] = len(char_vocab.keys()) # ----> confirm!! should also handle puncts properly

pickle.dump(char_vocab, open(MODEL_NAME_PREFIX + "_char_vocab.vocab", 'wb'))
print("Dumped vocab")

char_codes = set(["Lu", "Ll", "Lt", "Lm", "Lo"])
number_codes = set(["Nd", "Nl", "No"])
punc_codes = set(["Pc", "Pc", "Ps", "Pe", "Pi", "Pf", "Po"])
sym_codes = set(["Sm", "Sc", "Sk", "So"])
sep_codes = set(["Zs", "Zl", "Zp"])
mark_codes = set(["Mn", "Mc", "Me"])
other_codes = set(["Cc", "Cs", "Cf", "Co", "Cn"])
word_cateogry_sets = [char_codes, number_codes, punc_codes, sym_codes, sep_codes, mark_codes, other_codes]

n_classes = len(NER_sets_dict.keys()) + 1
n_chars = len(char_vocab.keys()) + 1

train_instances = []
test_instances = []
train_labels = []
test_labels = []

label_index_dict = {}

def get_one_hot_char(n_chars, char_id):
    assert char_id <= n_chars, "Invalid char id"
    char_vect = [0 for i in range(n_chars + 1)]
    char_vect[char_id] = 1
    return char_vect

def get_instance_from_word(word):
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
        for category_set in word_cateogry_sets:
            if character_category in category_set:
                c_vec.append(1)
            else:
                c_vec.append(0)
        ret_vecs.append(c_vec)
    return(ret_vecs)

max_len = 0
for tag_index, ner_tag in enumerate(NER_sets_dict.keys()):
    label_index_dict[ner_tag] = tag_index
    ner_words = NER_sets_dict[ner_tag]
    tag_instances = []
    tag_labels = []
    for word in ner_words:
        # tag_instances.append([char_vocab[char] for char in word])
        tag_instances.append(get_instance_from_word(word))
        zero_label_vect = [0 for i in range(n_classes)]
        zero_label_vect[tag_index] = 1
        assert tag_index < len(zero_label_vect) -1, "Non ner class label given wrongly"
        tag_labels.append(zero_label_vect)
        if len(word) > max_len:
            max_len = len(word)
    split_point = int(len(tag_instances) * (1-TEST_SPLIT))
    train_instances.extend(tag_instances[:split_point])
    train_labels.extend(tag_labels[:split_point])
    test_instances.extend(tag_instances[split_point:])
    test_labels.extend(tag_labels[split_point:])

non_ner_instances = []
non_ner_labels = []
for word in non_NER_set:
    # non_ner_instances.append([char_vocab[char] for char in word])
    non_ner_instances.append(get_instance_from_word(word))
    zero_label_vect = [0 for i in range(n_classes)]
    zero_label_vect[-1] = 1
    non_ner_labels.append(zero_label_vect)
    if len(word) > max_len:
        max_len = len(word)

split_point = int(len(non_ner_instances) * (1 - TEST_SPLIT))
train_instances.extend(non_ner_instances[:split_point])
train_labels.extend(non_ner_labels[:split_point])
test_instances.extend(non_ner_instances[split_point:])
test_labels.extend(non_ner_labels[split_point:])

"""
train_instances = np.array(train_instances)
rev_train_instances = np.copy(train_instances)

train_labels = np.array(train_labels)

test_instances = np.array(test_instances)
rev_test_instances = np.copy(test_instances)
"""

rev_train_instances = list(train_instances)
rev_test_instances = list(test_instances)

vec_len = len(train_instances[0][0])


for index, instance in enumerate(train_instances):
    zeros = [[0 for j in range(vec_len)] for i in range(max_len - len(instance))]
    train_instances[index] = zeros + instance

for index, instance in enumerate(rev_train_instances):
    zeros = [[0 for j in range(vec_len)] for i in range(max_len - len(instance))]
    rev_train_instances[index] = zeros + instance[::-1]

for index, instance in enumerate(test_instances):
    zeros = [[0 for j in range(vec_len)] for i in range(max_len - len(instance))]
    test_instances[index] = zeros + instance

for index, instance in enumerate(rev_test_instances):
    zeros = [[0 for j in range(vec_len)] for i in range(max_len - len(instance))]
    rev_test_instances[index] = zeros + instance[::-1]

train_instances = np.array(train_instances)
train_labels = np.array(train_labels)

rev_train_instances = np.array(rev_train_instances)

test_instances = np.array(test_instances)
test_labels = np.array(test_labels)

rev_test_instances = np.array(rev_test_instances)

print("Train input shapes")
print(train_instances.shape)
print(rev_train_instances.shape)

print("Test input shapes")
print(test_instances.shape)
print(rev_test_instances.shape)

# CONFIRM IF INPUTS ARE CORRECT!

N_BATCH = 2048
N_EPOCH = 100

EMBEDDING_DROPOUT = 0.1
EMBEDDING_DIM = 60
LSTM_OUTPUT_DIM = 100
PRE_PREDICT_DIM = 100
LSTM_REG_WEIGHT = 0.01
PRE_PRED_REG = 0.01
PRED_REG = 0.01

"""
Try using lstm output at each sequence and convolving over it.
Also try grouping datasets based on length of sequence for training.
"""


print("Creating model")
fixed_char_encoder = Graph()

fixed_char_encoder.add_input("forward_input", (train_instances.shape[1],train_instances.shape[2])) # train_instances.shape[2] vs. vec_size?
fixed_char_encoder.add_input("backward_input", (rev_train_instances.shape[1],rev_train_instances.shape[2]))

# fixed_char_encoder.add_input("forward_embed", (train_instances.shape[1],train_instances.shape[2]), dtype='int') # train_instances.shape[2] vs. vec_size?
# fixed_char_encoder.add_input("backward_embed", (rev_train_instances.shape[1],rev_train_instances.shape[2]), dtype='int')

print("Added inputs")

# forward_embedding_layer = Embedding(input_dim=n_chars, output_dim=EMBEDDING_DIM, mask_zero=True, dropout=EMBEDDING_DROPOUT)
# backward_embedding_layer = Embedding(input_dim=n_chars, output_dim=EMBEDDING_DIM, mask_zero=True, dropout=EMBEDDING_DROPOUT)
# fixed_char_encoder.add_node(forward_embedding_layer, input="forward_input", name="forward_embed")
# fixed_char_encoder.add_node(backward_embedding_layer, input="backward_input", name="backward_embed")

# Using time distributed dense instead of embedding layer
forward_embed = TimeDistributedDense(output_dim=EMBEDDING_DIM, weights=None,W_regularizer=None,b_regularizer=None,input_shape=(train_instances.shape[1], train_instances.shape[2]))
backward_embed = TimeDistributedDense(output_dim=EMBEDDING_DIM, weights=None,W_regularizer=None,b_regularizer=None,input_shape=(rev_train_instances.shape[1], rev_train_instances.shape[2]))
fixed_char_encoder.add_node(forward_embed, input="forward_input", name="forward_embed")
fixed_char_encoder.add_node(backward_embed, input="backward_input", name="backward_embed")

print("Added embedding layers")

forward_LSTM = LSTM(output_dim=LSTM_OUTPUT_DIM,dropout_W=0.1, dropout_U=0.1, W_regularizer=l2(LSTM_REG_WEIGHT), return_sequences=False) # Try by pooling sequence of outputs as well
backward_LSTM = LSTM(output_dim=LSTM_OUTPUT_DIM,dropout_W=0.1, dropout_U=0.1, W_regularizer=l2(LSTM_REG_WEIGHT), return_sequences=False) # Try by pooling sequence of outputs as well

fixed_char_encoder.add_node(forward_LSTM, input="forward_embed", name="forward_lstm")
fixed_char_encoder.add_node(backward_LSTM, input="backward_embed", name="backward_lstm")

print("Added LSTM layers")

pre_predict_layer = Dense(output_dim=PRE_PREDICT_DIM, activation="tanh", W_regularizer=l2(PRE_PRED_REG))

fixed_char_encoder.add_node(pre_predict_layer, name="pre_predict", inputs=["forward_lstm", "backward_lstm"], merge_mode="concat")

print("Added pre_predict layer")

predict_layer = Dense(output_dim=n_classes, activation="softmax", W_regularizer=l2(PRED_REG))
fixed_char_encoder.add_node(predict_layer, name="predict_output", input="pre_predict")

print("Added predict layer")

fixed_char_encoder.add_output(name="ner_output",input="predict_output")

print("Added output. Now compiling model")

# fixed_char_encoder.compile(optimizer="rmsprop", loss={"ner_output":'categorical_crossentropy'})
fixed_char_encoder.compile(optimizer=RMSprop(lr=0.001, rho=0.5, epsilon=1e-06), loss={"ner_output":'categorical_crossentropy'})

print("Compiled model. Now fitting data")

history = fixed_char_encoder.fit(data={"forward_input":train_instances, "backward_input":rev_train_instances, "ner_output":train_labels}, batch_size=N_BATCH, nb_epoch=N_EPOCH, validation_split=0.1)

print("Predicting with test set:")
print("test_instances of shape:", test_instances.shape, rev_test_instances.shape)

predictions = fixed_char_encoder.predict({"forward_input":test_instances, "backward_input":rev_test_instances}) # {'output':...}

print("Obtained predictions")
# print(predictions)

ner_outputs = predictions['ner_output']

print("Obtained ner_outputs")
print(ner_outputs)
print("Shape of ner_outputs:", ner_outputs.shape)

print("Saving model architecture")

with open(MODEL_NAME_PREFIX + str(N_EPOCH) + "_" + str(N_BATCH) +  ".arch", 'wb') as model_file:
    model_file.write(fixed_char_encoder.to_json())

print("Saving model weights")

fixed_char_encoder.save_weights(MODEL_NAME_PREFIX + str(N_EPOCH) + "_" + str(N_BATCH) + ".weights.h5", overwrite=True)

print("Outputs shape: ", ner_outputs.shape)
print("Test labels shape: ", test_labels.shape)
print("Test instances shape: ", test_instances.shape, rev_test_instances.shape)

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

accuracies, test_conf_mat = get_accuracies(ner_outputs, test_labels)

print("Test stats")

print(accuracies)
print(test_conf_mat)

predictions = fixed_char_encoder.predict({"forward_input":train_instances, "backward_input":rev_train_instances}) # {'output':...}

ner_outputs = predictions['ner_output']

accuracies, train_conf_mat = get_accuracies(ner_outputs, train_labels)

print("Train stats")

print(accuracies)
print(train_conf_mat)

print label_index_dict

raw_input("Enter to end")