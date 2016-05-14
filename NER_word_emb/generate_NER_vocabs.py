from my_utils import *

TRAIN_CONLL = "./conll_data/turkish_conll_train.conll"
TEST_CONLL = "./conll_data/turkish_conll_test.conll"
DEV_CONLL = "./conll_data/turkish_conll_dev.conll"

NER_VOCAB_PREFIX = "Turkish_NER_Vocab_"
NON_NER_VOCAB_PREFIX = "Turkish_NON_NER_Vocab"


# TRAIN_CONLL = "./conll_data/uzbek_conll_train.conll"
# TEST_CONLL = "./conll_data/uzbek_conll_test.conll"
# DEV_CONLL = "./conll_data/uzbek_conll_dev.conll"

# NER_VOCAB_PREFIX = "Uzbek_NER_Vocab_"
# NON_NER_VOCAB_PREFIX = "Uzbek_NON_NER_Vocab"

def generate_ner_vocabs():
    vocab, NER_sets_dict, non_NER_set, wlen_dicts = get_all_words_charset_and_ner_sets(file_list=[TRAIN_CONLL, DEV_CONLL,
                                                                                         TEST_CONLL],
                                                                                       word_len_threshold=sys.maxint,
                                                                                        ignore_words=False)
    for NER_TAG in NER_sets_dict.keys():
        with open(NER_VOCAB_PREFIX + str(NER_TAG) + ".txt", 'w') as op_file:
            for NER in NER_sets_dict[NER_TAG]:
                op_file.write(NER.encode("utf-8") + "\n")
    with open(NON_NER_VOCAB_PREFIX + ".txt", 'w') as op_file:
        for word in non_NER_set:
                op_file.write(word.encode("utf-8") + "\n")

generate_ner_vocabs()