from keras.layers import containers
from keras.models import Sequential, Graph
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedMerge, AutoEncoder, Merge, RepeatVector, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.regularizers import l1,l2
from keras.optimizers import RMSprop
import numpy as np

TRAIN_CONLL = "./conll_data/turkish_conll_train.conll"
TEST_CONLL = "./conll_data/turkish_conll_test.conll"
DEV_CONLL = "./conll_data/turkish_conll_dev.conll"


TEST_SPLIT = 0.1

char_vocab = set()
char_vocab.update()
NER_sets_dict = {}
non_NER_set = set()

def get_words_charset_and_ner_sets(vocab, ner_sets_dict, non_ner_set, file):
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
            vocab.update(list(word))
    return(vocab, ner_sets_dict, non_ner_set)

char_vocab, NER_sets_dict, non_NER_set = get_words_charset_and_ner_sets(char_vocab, NER_sets_dict, non_NER_set, TRAIN_CONLL)
char_vocab, NER_sets_dict, non_NER_set = get_words_charset_and_ner_sets(char_vocab, NER_sets_dict, non_NER_set, DEV_CONLL)
char_vocab, NER_sets_dict, non_NER_set = get_words_charset_and_ner_sets(char_vocab, NER_sets_dict, non_NER_set, TEST_CONLL)

char_vocab = {char : char_index + 1 for char_index, char in enumerate(char_vocab)}

n_classes = len(NER_sets_dict.keys()) + 1
n_chars = len(char_vocab.keys()) + 1

train_instances = []
test_instances = []
train_labels = []
test_labels = []

label_index_dict = {}

max_len = 0
for tag_index, ner_tag in enumerate(NER_sets_dict.keys()):
    label_index_dict[ner_tag] = tag_index
    ner_words = NER_sets_dict[ner_tag]
    tag_instances = []
    tag_labels = []
    for word in ner_words:
        tag_instances.append([char_vocab[char] for char in word])
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
    non_ner_instances.append([char_vocab[char] for char in word])
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

deb_train = train_labels[-100:]
deb_test = test_labels[-100:]

"""
train_instances = np.array(train_instances)
rev_train_instances = np.copy(train_instances)

train_labels = np.array(train_labels)

test_instances = np.array(test_instances)
rev_test_instances = np.copy(test_instances)
"""

rev_train_instances = list(train_instances)
rev_test_instances = list(test_instances)
print(len(test_instances))
print(len(rev_test_instances))


for index, instance in enumerate(train_instances):
    zeros = [0 for i in range(max_len - len(instance))]
    train_instances[index] = zeros + instance

for index, instance in enumerate(rev_train_instances):
    zeros = [0 for i in range(max_len - len(instance))]
    rev_train_instances[index] = zeros + instance[::-1]

for index, instance in enumerate(test_instances):
    zeros = [0 for i in range(max_len - len(instance))]
    test_instances[index] = zeros + instance

for index, instance in enumerate(rev_test_instances):
    zeros = [0 for i in range(max_len - len(instance))]
    rev_test_instances[index] = zeros + instance[::-1]

train_instances = np.array(train_instances)
train_labels = np.array(train_labels)

rev_train_instances = np.array(rev_train_instances)

test_instances = np.array(test_instances)
test_labels = np.array(test_labels)

rev_test_instances = np.array(rev_test_instances)

print("Test input shapes")
print(test_instances.shape)
print(rev_test_instances.shape)

# CONFIRM IF INPUTS ARE CORRECT!

EMBEDDING_DROPOUT = 0.1
EMBEDDING_DIM = 50
LSTM_OUTPUT_DIM = 100
PRE_PREDICT_DIM = 100
LSTM_REG_WEIGHT = 0.01
PRE_PRED_REG = 0.01
PRED_REG = 0.01

print("Creating model")
fixed_char_encoder = Graph()

fixed_char_encoder.add_input("forward_input", (train_instances.shape[1],), dtype='int')
fixed_char_encoder.add_input("backward_input", (rev_train_instances.shape[1],), dtype='int')

print("Added inputs")

forward_embedding_layer = Embedding(input_dim=n_chars, output_dim=EMBEDDING_DIM, mask_zero=True, dropout=EMBEDDING_DROPOUT)
backward_embedding_layer = Embedding(input_dim=n_chars, output_dim=EMBEDDING_DIM, mask_zero=True, dropout=EMBEDDING_DROPOUT)
fixed_char_encoder.add_node(forward_embedding_layer, input="forward_input", name="forward_embed")
fixed_char_encoder.add_node(backward_embedding_layer, input="backward_input", name="backward_embed")

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

history = fixed_char_encoder.fit(data={"forward_input":train_instances, "backward_input":rev_train_instances, "ner_output":train_labels}, batch_size=128, nb_epoch=20, validation_split=0.1)

predictions = fixed_char_encoder.predict({"forward_input":test_instances, "backward_input":rev_test_instances}) # {'output':...}

ner_outputs = predictions['ner_output']

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