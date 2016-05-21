import pylab as pl
import sys
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-conll", "--conll_file", help="Source CONLL file",
                        type=str)
arg_parser.add_argument("-op", "--output_file", help="Output plain text file name",
                        type=str)
arg_parser.add_argument("-ner_thresh", "--ner_rare_threshold", help="Rare threshold for named entities", default=0,
                        type=int)
arg_parser.add_argument("-non_ner_thresh", "--non_ner_rare_threshold", help="Rare threshold for non named entities",
                        default=0, type=int)
arg_parser.add_argument("-ner_unk", "--ner_rare_token", help="Rare token for named entities", default="NER_UNK",
                        type=str)
arg_parser.add_argument("-non_ner_unk", "--non_ner_rare_token", help="Rare token for non named entities",
                        default="NON_NER_UNK", type=str)


args = arg_parser.parse_args()
print("Args used for this run:")
print(args)

# CONLL_FILE = "./conll_generated_files/uzbek_all_suffix_exclude.conll"
CONLL_FILE = args.conll_file

# OP_FILE = "./conll_generated_files/uzbek_ner_thresh_" + str(NER_THRESH) + "_nonner_thresh_" + str(NON_NER_THRESH) + "_suffix_exclude.txt"
OP_FILE = args.output_file

# NER_THRESH = 5
# NON_NER_THRESH = 10

NER_THRESH = args.ner_rare_threshold
NON_NER_THRESH = args.non_ner_rare_threshold

NON_NER_UNK = args.non_ner_rare_token
NER_UNK = args.ner_rare_token

# NON_NER_UNK = "NON_NER_UNK"
# NER_UNK = "NER_UNK"

def get_vocab_counts(input_file):
    counts_dict = {"NER" : {}, "NON-NER": {}}
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
    return(counts_dict)

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

counts_dict = get_vocab_counts(CONLL_FILE)
# print("For NERs")
# visualize(counts_dict["NER"])
# print("For non-NERs")
# visualize(counts_dict["NON-NER"])

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

ner_unks = unkify(counts_dict["NER"], NER_THRESH)
non_ner_unks = unkify(counts_dict["NON-NER"], NON_NER_THRESH)

with open(CONLL_FILE, "r") as ip_file:
    op_str = ""
    for line in ip_file:
        line = line.strip(" \r\t\n")
        if len(line) == 0:
            continue
        else:
            word, tag = line.split("\t")

        unkified_word = get_unk(word, ner_unks=ner_unks, non_ner_unks=non_ner_unks)
        try:
            op_str += get_unk(word, ner_unks=ner_unks, non_ner_unks=non_ner_unks) + " "
        except:
            print("Problem")


with open(OP_FILE, "a") as op_file:
    op_file.write(op_str)