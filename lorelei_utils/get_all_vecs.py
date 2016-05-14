import pylab as pl
import sys

# CONLL_FILE = "./conll_generated_files/turkish_splits/turkish_all.conll" # For include suffix case
CONLL_FILE = "./conll_generated_files/turkish_splits/turkish_suffix_exclude_all.conll"

# VEC_FILE = "./conll_generated_files/vecs/turkish/turkish_ner_thresh_2_nonner_thresh_2_vecs.txt" # For include suffix case
VEC_FILE = "./conll_generated_files/vecs/turkish/turkish_ner_thresh_9223372036854775807_nonner_thresh_5_suffix_exclude_vecs.txt"

NER_THRESH = 9223372036854775807
NON_NER_THRESH = 5

# OP_FILE = "./conll_generated_files/vecs/turkish/turkish_ner_thresh_2_nonner_thresh_2_final_vecs.txt" # for include suffix case
OP_FILE = "./conll_generated_files/vecs/turkish/turkish_ner_thresh_9223372036854775807_nonner_thresh_5_suffix_exclude_final_vecs.txt"


NON_NER_UNK = "NON_NER_UNK"
NER_UNK = "NER_UNK"

def get_vocab_counts(input_file):
    counts_dict = {"NER" : {}, "NON-NER": {}}
    vocab = set()
    NER_Set = set(['B-PER', 'B-LOC', 'B-ORG', 'B-TTL', 'I-PER', 'I-LOC', 'I-ORG', 'I-TTL'])
    with open(input_file, "r") as ip_file:
        for line in ip_file:
            line = line.strip(" \t\r\n")
            if len(line) == 0:
                continue
            word, ner_tag = [val.strip("\t\r\n ") for val in line.split("\t")]
            if ner_tag in NER_Set:
                counts_dict["NER"][word] = counts_dict["NER"].get(word, 0) + 1
            else:
                counts_dict["NON-NER"][word] = counts_dict["NON-NER"].get(word, 0) + 1
            vocab.add(word)
    return(counts_dict, vocab)

def visualize(counts_dict, rare_thresholds=None):
    counts_to_n_words_dict = {}
    for word in counts_dict.keys():
        counts_to_n_words_dict[counts_dict[word]] = counts_to_n_words_dict.get(counts_dict[word], 0) + 1
    tot_words = len(counts_dict.keys())
    if rare_thresholds is None:
        median_count = list(counts_to_n_words_dict.keys())
        median_count.sort()
        median_count = median_count[len(median_count) / 2]
        rare_thresholds = [int(percent * 0.01 * median_count) for percent in
                           [0.01, 0.1, 0.5, 1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500]]
        for index, rare_threshold in enumerate(rare_thresholds):
            if rare_threshold == 0:
                del rare_thresholds[index]  # removing rare flickr_thresholds of 0
        rare_thresholds.sort()

        rare_elim_counts = {rare_threshold: 0 for rare_threshold in rare_thresholds}
        # dict giving percentage of words eliminated based on a given rare_threshold

        for count in counts_to_n_words_dict.keys():
            for rare_threshold in rare_elim_counts.keys():
                if count < rare_threshold:
                    rare_elim_counts[rare_threshold] += counts_to_n_words_dict[count]

        for rare_threshold in rare_thresholds:
            print("Rare threshold = %f, percentage eliminated is %f and no. of actual words eliminated is %f" % (
                rare_threshold, rare_elim_counts[rare_threshold] * 1.0 / tot_words, rare_elim_counts[rare_threshold]))
            print("Tot no. of words is %d" % (tot_words))

        d = counts_to_n_words_dict
        # X = np.arange(len(d))
        X = d.keys()
        X.sort(reverse=True)
        Y = [d[x] for x in X]
        pl.bar(X, Y, align='center', width=0.5)
        pl.xticks(X, X)
        ymax = max(Y) + 1
        pl.ylim(0, ymax)
        pl.show()

def get_vecs(vec_file):
    vec_dict = {}
    with open(vec_file, 'r') as ip_file:
        vocab_size, vocab_dim = [int(val) for val in ip_file.readline().strip(" \t\r\n").split()]
        for line in ip_file:
            line = line.strip(" \t\r\n").split()
            word = line[0]
            w_vec = [float(dim) for dim in line[1:]]
            assert len(w_vec) == vocab_dim, "Wrong vector!"
            vec_dict[word] = w_vec
    return(vec_dict)

def unkify(word_counts_dict, threshold=sys.maxint):
    rare_set = set([])
    for word in word_counts_dict.keys():
        if word_counts_dict[word] < threshold:
            rare_set.add(word)
    return(rare_set)

def get_unk(word, ner_unks=set(), non_ner_unks=set()):
    if word in ner_unks:
        return(NER_UNK)
    elif word in non_ner_unks:
        return(NON_NER_UNK)
    else:
        return(word)

counts_dict, vocab = get_vocab_counts(CONLL_FILE)
w2vec = get_vecs(VEC_FILE)
# print("For NERs")
# visualize(counts_dict["NER"])
# print("For non-NERs")
# visualize(counts_dict["NON-NER"])

ner_unks = unkify(counts_dict["NER"], NER_THRESH)
non_ner_unks = unkify(counts_dict["NON-NER"], NON_NER_THRESH)

start_sym = "</s>"

final_vecs = {start_sym: w2vec[start_sym]}


for word in vocab:
    assert get_unk(word, ner_unks, non_ner_unks) in w2vec.keys(), "Truly unknown word!"
    final_vecs[word] = w2vec[get_unk(word, ner_unks, non_ner_unks)]


with open(OP_FILE, 'w') as op_file:
    op_file.write(str(len(final_vecs.keys())) + " " + str(len(final_vecs[word])) + "\n")
    op_file.write(start_sym + " " + " ".join([str(dim) for dim in final_vecs[start_sym]]) + "\n")
    for word in vocab:
        op_file.write(word + " ".join([str(dim) for dim in final_vecs[word]]) + "\n")

