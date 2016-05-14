# -*- coding: utf-8 -*-
from nltk.stem.snowball import SnowballStemmer
import random
import pickle
from collections import Counter


"""

MAY BE WORTH EXTENDING THIS TO BE ABLE TO 'TUNE' PARAMETERS ON A NEW DATASET USING SEED PARAMETERS OBTAINED FROM A
DIFFERENT DATASET.

"""





class PoorManStemmer():
    def __init__(self, prefix_len):
        self.prefix_len = prefix_len

    def stem(self, word): # Receives a utf decoded string and returns a utf decoded string
        return(word[:self.prefix_len])
        # utf_word = word#.decode('utf-8')
        # utf_stem = utf_word[:self.prefix_len]
        # enc_stem = utf_stem#.encode('utf-8')
        # return(enc_stem)



class Model1(object):
    german_stemmer = SnowballStemmer("german")
    english_stemmer = SnowballStemmer("english")
    poor_man_stemmer = PoorManStemmer(prefix_len=5)
    stemmers = {"german": german_stemmer, "english": english_stemmer}

    null_val = ''.decode('utf-8').encode('utf-8')
    rare_token = "__RARE_TOKEN_"

    #Both vocabularies have a null
    foreign_vocab = set([null_val])
    source_vocab = set([null_val])

    #Conditional probabilities are initialized to a uniform distribution over all german words conditioned on an english
    #word. However, most words never co-occurr. To save space and avoid storing such parameters that are bound to be 0, we
    #only store those conditional probabilities involving word pairs that actually co-occur.

    translation_probs = {}

    #Expected count number of alignments between each german word - english word pair

    counts = {}

    #Expected number of all alignments involving an english word (sum counts over all german words while fixing the english
    #word

    totals = {}

    rare_tokens = (set(), set()) # Set of tokens clubbed as rare for foreign and source language respectively

    def __init__(self, parameter_file=None, poor_man_stem_length=5, foreign_language="german", source_language="english"):
        # Input file expected in the format <foreign lang sentence> ||| <source lang sentence>
        if parameter_file is not None:
            (self.translation_probs,self.rare_tokens) = pickle.load(open(parameter_file,'rb'))
        self.INPUT_FILE = None
        self.poor_man_stemmer = PoorManStemmer(poor_man_stem_length)
        self.foreign_language = foreign_language
        self.source_language = source_language

    def get_params(self):
        return(self.translation_probs)

    def get_german_stem(self, word):
        return(self.german_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8'))

    def get_english_stem(self, word):
        return(self.english_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8'))

    def get_poor_man_stem(self, word):
        return(self.poor_man_stemmer.stem(word))

    def get_stem(self, word, language): # accepts un-decoded string and returns encoded(=undecoded) string
        stemmer = self.stemmers.get(language, self.poor_man_stemmer)
        return(stemmer.stem(word.strip().decode('utf-8')).encode('utf-8'))

    def get_translation_prob(self, foreign_stem, source_stem):
        return self.translation_probs.get((foreign_stem, source_stem), 0.0)

    def get_parallel_instance(self, corpus_line):
        # Assumes each line in corpus is of the form: <foregn lang sentence> ||| <source lang sentence>
        try:
            [foreign, source] = corpus_line.strip().split(' ||| ')
        except:
            print(corpus_line)
            raw_input("Enter to continue")
        foreign_stemmed_sentence,source_stemmed_sentence = ([self.get_stem(word, self.foreign_language).lower() for word in foreign.split(' ')],
                                                            [self.get_stem(word, self.source_language).lower() for word in source.split(' ')])
            # ([self.get_german_stem(word).lower() for word in foreign.split(' ')],
            #  [self.get_english_stem(word).lower() for word in source.split(' ')])

        return ([self.rare_token if tok in self.rare_tokens[0] else tok
                 for tok in foreign_stemmed_sentence],
                [self.rare_token if tok in self.rare_tokens[1] else tok
                 for tok in source_stemmed_sentence])

    def get_counts(self, sent):
        """
        Returns a dicts mapping stemmed words to
        their respective counts in the sentence sent.
        Also, one null token count is added to sentence
        """
        ret = {self.null_val:1}
        for word in sent:
            ret[word] = ret.get(word, 0) + 1
        return ret

    def get_prior(self,**kwargs):
        return 1

    def get_alignment(self, german, english):
        """
        Returns model1 alignment for a DE/EN parallel sentence pair.
        For each german word, identifies
        the best english word (or NULL) to align to
        """
        english.append(self.null_val)
        alignment = []
        for (i, g_i) in enumerate(german):
            best = -1
            bestscore = 0
            for (j, e_j) in enumerate(english):
                val = self.get_prior()*self.get_translation_prob(g_i,e_j)
                if best==-1 or val>bestscore:
                    bestscore = val
                    best = j
            if best < len(english)-1:
                yield (i,best) # don't yield anything for NULL alignment



class EM_model1(Model1):

    SOURCE_TO_FOREIGN = 1
    FOREIGN_TO_SOURCE = 2

    source_to_foreign_translation_probs = {}
    foreign_to_source_translation_probs = {}

    foreign_totals = {}
    source_totals = {}

    RARE_THRESHOLD = 2
    direction = None

    def get_params(self, direction):
        assert direction in [self.SOURCE_TO_FOREIGN, self.FOREIGN_TO_SOURCE], "Invalid translation direction"
        # return(self.foreign_to_source_translation_probs, self.source_to_foreign_translation_probs)
        if direction == self.SOURCE_TO_FOREIGN:
            return(self.source_to_foreign_translation_probs)
        elif direction == self.FOREIGN_TO_SOURCE:
            return(self.foreign_to_source_translation_probs)

    def get_translation_prob(self, foreign_stem, source_stem):
        assert self.direction is not None, "First estimate params"
        if self.direction == self.SOURCE_TO_FOREIGN:
            return (self.source_to_foreign_translation_probs[(foreign_stem, source_stem)])
        elif self.direction == self.FOREIGN_TO_SOURCE:
            return (self.foreign_to_source_translation_probs[(source_stem, foreign_stem)])



    def __init__(self, input_file, output_file, n_iterations, rare_threshold=2, source_language="english", foreign_language="german"):
        self.MAX_ITERS = n_iterations
        self.INPUT_FILE = input_file
        self.OUTPUT_FILE = output_file
        self.RARE_THRESHOLD = rare_threshold
        self.source_language=source_language
        self.foreign_language=foreign_language
        self.rare_tokens = (set(),set())
        self.rare_tokens = self._find_rares()

    def store_rares(self, op_file):
        pickle.dump(self.rare_tokens, open(op_file, 'wb'))

    def _find_rares(self):
        source_counts = Counter()#{}
        foreign_counts = Counter()
        with open(self.INPUT_FILE) as ip:
            for line in ip:
                [foreign_stemmed_sentence, source_stemmed_sentence] = super(EM_model1,self).get_parallel_instance(line)
                foreign_counts.update(foreign_stemmed_sentence)
                source_counts.update(source_stemmed_sentence)
                #for stem in source_stemmed_sentence:
                #    if stem not in counts:
                #        counts[stem] = Counter(foreign_stemmed_sentence)
                #    else:
                #        counts[stem].update(source_stemmed_sentence)
        source_rares = set([word for word in source_counts if source_counts[word]<self.RARE_THRESHOLD])
        foreign_rares = set([word for word in foreign_counts if foreign_counts[word]<self.RARE_THRESHOLD])
        return (foreign_rares,source_rares)

    def preprocess(self, direction):
        assert direction == self.SOURCE_TO_FOREIGN or direction == self.FOREIGN_TO_SOURCE, "Invalid translation direction"
        ip_line_counter = 0
        corpus = []
        with open(self.INPUT_FILE) as ip:
            print("Starting to process corpus")
            for line in ip:
                ip_line_counter += 1
                if (ip_line_counter % 1000 == 0):
                    print("Processed %d lines" % (ip_line_counter))
                [foreign_stemmed_sentence, source_stemmed_sentence] = self.get_parallel_instance(line)
                foreign_word_counts = {}
                source_word_counts = {}
                for foreign_stemmed_word in foreign_stemmed_sentence:
                    foreign_word_counts[foreign_stemmed_word] = foreign_word_counts.get(foreign_stemmed_word, 0.0) + 1.0
                    self.foreign_vocab.add(foreign_stemmed_word)
                    if direction == self.FOREIGN_TO_SOURCE:
                        for source_stemmed_word in source_stemmed_sentence:
                            key = (source_stemmed_word, foreign_stemmed_word)
                            if not self.foreign_to_source_translation_probs.has_key(key):
                                self.foreign_totals[foreign_stemmed_word] = self.foreign_totals.get(
                                    foreign_stemmed_word, 0.0) + 1.0
                            self.foreign_to_source_translation_probs[key] = 1.0
                for source_stemmed_word in source_stemmed_sentence:
                    source_word_counts[source_stemmed_word] = source_word_counts.get(source_stemmed_word, 0.0) + 1.0
                    self.source_vocab.add(source_stemmed_word)
                    if direction == self.SOURCE_TO_FOREIGN:
                        for foreign_stemmed_word in foreign_stemmed_sentence:
                            key = (foreign_stemmed_word, source_stemmed_word)
                            if not self.source_to_foreign_translation_probs.has_key(key):
                                self.source_totals[source_stemmed_word] = self.source_totals.get(
                                    source_stemmed_word, 0.0) + 1.0
                            self.source_to_foreign_translation_probs[key] = 1.0
                foreign_word_counts[self.null_val] = 1  # null added to sentence
                source_word_counts[self.null_val] = 1  # null added to sentence
                corpus.append([foreign_word_counts, source_word_counts])
        return (corpus)

    def normalize(self, direction):
        for source_stemmed_word in self.source_vocab:
            for foreign_stemmed_word in self.foreign_vocab:
                if direction == self.SOURCE_TO_FOREIGN:
                    key = (foreign_stemmed_word, source_stemmed_word)
                    if self.source_to_foreign_translation_probs.has_key(
                            key):  # prevent populating entries unless they occur in parallel sentences
                        self.source_to_foreign_translation_probs[key] = self.source_to_foreign_translation_probs[key] / \
                                                                        self.source_totals[
                                                                            source_stemmed_word]  # source_totals of english word should NEVER be 0
                    elif source_stemmed_word == self.null_val:
                        self.source_to_foreign_translation_probs[key] = 1.0 / len(self.foreign_vocab)
                elif direction == self.FOREIGN_TO_SOURCE:
                    key = (source_stemmed_word, foreign_stemmed_word)
                    if self.foreign_to_source_translation_probs.has_key(
                            key):  # prevent populating entries unless they occur in parallel sentences
                        self.foreign_to_source_translation_probs[key] = self.foreign_to_source_translation_probs[key] / \
                                                                        self.foreign_totals[
                                                                            foreign_stemmed_word]  # foreign_totals of german word should NEVER be 0
                    elif foreign_stemmed_word == self.null_val:
                        self.foreign_to_source_translation_probs[key] = 1.0 / len(self.source_vocab)

    def stop_condition(self,
                       iter_count):  # Currently only checking for iteration limit. Ideally, we should also check for
        # convergence, i.e., when parameters change by value below a certain threshold
        if iter_count == self.MAX_ITERS:
            return(True)
        else:
            return(False)

    def estimate_params(self, direction, store_frequency):
        assert direction == self.FOREIGN_TO_SOURCE or direction == self.SOURCE_TO_FOREIGN, "Invalid direction specified"

        # May have to take care of last line being empty
        self.corpus = self.preprocess(direction)
        self.direction = direction

        self.normalize(direction)

        iter_count = 0

        """
        EM algorithm for estimating the translation probablities
        See https://www.cl.cam.ac.uk/teaching/1011/L102/clark-lecture3.pdf for a good tutorial
        """

        while(True):  #until convergence or max_iters
            print("Iteration " + str(iter_count + 1))
            iter_count += 1
            self.counts = {}  # All counts default to 0. These are counts of (german, english) word pairs
            self.source_totals = {}  # All source_totals default to 0. These are sums of counts (marginalized over all foreign words), for each
            self.foreign_totals = {}  # totals for german words, used when estimating p(english word | foreign_word) instead of p(foreign_word | source_word)
            for parallel_instance in self.corpus:  # Stemmed parallel instances stored in memory to speed up EM
                foreign_sent_dict = parallel_instance[0]
                source_sent_dict = parallel_instance[1]
                if direction == self.SOURCE_TO_FOREIGN:
                    for foreign_word in foreign_sent_dict.keys():  # For each unique german word in the german sentence
                        foreign_word_count = foreign_sent_dict[foreign_word]  # Its count in the german sentence
                        total_s = 0.0  # Expected count of number of alignments for this german word with any english word
                        for source_word in source_sent_dict.keys():
                            total_s += self.source_to_foreign_translation_probs.get((foreign_word, source_word),
                                                                                    0.0) * foreign_word_count
                        for source_word in source_sent_dict.keys():
                            source_word_count = source_sent_dict[source_word]
                            if self.counts.has_key(source_word):
                                self.counts[source_word][foreign_word] = self.counts[source_word].get(foreign_word,
                                                                                                       0.0) + self.source_to_foreign_translation_probs.get(
                                    (foreign_word, source_word), 0.0) * foreign_word_count * source_word_count / total_s
                            else:
                                self.counts[source_word] = {}
                                self.counts[source_word][foreign_word] = self.counts[source_word].get(foreign_word,
                                                                                                       0.0) + self.source_to_foreign_translation_probs.get(
                                    (foreign_word, source_word), 0.0) * foreign_word_count * source_word_count / total_s
                            # Expected count of alignments between german word and this english word, divided by the expected count of all alignments of this german word
                            self.source_totals[source_word] = self.source_totals.get(source_word,
                                                                                      0.0) + self.source_to_foreign_translation_probs.get(
                                (foreign_word, source_word), 0.0) * foreign_word_count * source_word_count / total_s
                            # Aggregating the expected counts of all german words, for each english word. This will be used as a normalizing factor
                elif direction == self.FOREIGN_TO_SOURCE:
                    for source_word in source_sent_dict.keys():  # For each unique german word in the german sentence
                        source_word_count = source_sent_dict[source_word]  # Its count in the german sentence
                        total_s = 0.0  # Expected count of number of alignments for this german word with any english word
                        for foreign_word in foreign_sent_dict.keys():
                            total_s += self.foreign_to_source_translation_probs.get((source_word, foreign_word),
                                                                                    0.0) * source_word_count
                        for foreign_word in foreign_sent_dict.keys():
                            foreign_word_count = foreign_sent_dict[foreign_word]
                            if self.counts.has_key(foreign_word):
                                self.counts[foreign_word][source_word] = self.counts[foreign_word].get(source_word,
                                                                                                      0.0) + self.foreign_to_source_translation_probs.get(
                                    (source_word, foreign_word), 0.0) * source_word_count * foreign_word_count / total_s
                            else:
                                self.counts[foreign_word] = {}
                                self.counts[foreign_word][source_word] = self.counts[foreign_word].get(source_word,
                                                                                                      0.0) + self.foreign_to_source_translation_probs.get(
                                    (source_word, foreign_word), 0.0) * source_word_count * foreign_word_count / total_s
                            # Expected count of alignments between german word and this english word, divided by the expected count of all alignments of this german word
                            self.foreign_totals[foreign_word] = self.foreign_totals.get(foreign_word,
                                                                                       0.0) + self.foreign_to_source_translation_probs.get(
                                (source_word, foreign_word), 0.0) * source_word_count * foreign_word_count / total_s
                            # Aggregating the expected counts of all german words, for each english word. This will be used as a normalizing factor
            if direction == self.SOURCE_TO_FOREIGN:
                for source_word in self.source_totals.keys():  # restricting to domain total( . )
                    for foreign_word in self.counts[source_word].keys():
                        self.source_to_foreign_translation_probs[(foreign_word, source_word)] = self.counts[
                                                                                                    source_word].get(
                            foreign_word, 0.0) / self.source_totals.get(source_word, 0.0)
                        # Neither domain nor counts should never be 0 given our domain restriction
            elif direction == self.FOREIGN_TO_SOURCE:
                for foreign_word in self.foreign_totals.keys():  # restricting to domain total( . )
                    for source_word in self.counts[foreign_word].keys():
                        self.foreign_to_source_translation_probs[source_word, foreign_word] = self.counts[
                                                                                                  foreign_word].get(
                            source_word, 0.0) / self.foreign_totals.get(foreign_word, 0.0)
                        # Neither domain nor counts should never be 0 given our domain restriction

            if (iter_count % store_frequency == 0):  # Store the model at some frequency of iterations
                print("Storing model after %d iterations" % (iter_count))
                model_dump = open(self.OUTPUT_FILE, 'wb')
                if direction == self.SOURCE_TO_FOREIGN:
                    # print("Spot checking on 5 percent of source %s 's vocabulary before storing!"%(self.source_language))
                    # self.sanity_check(direction, int(len(self.source_vocab) * 0.05))
                    pickle.dump((self.source_to_foreign_translation_probs, self.rare_tokens), model_dump)
                    print("Storing source language %s to foreign language %s translation model after %d iterations" % (self.source_language, self.foreign_language, iter_count))
                elif direction == self.FOREIGN_TO_SOURCE:
                    # print("Spot checking on 5 percent of foreign language %s vocabulary before storing!"%(self.foreign_language))
                    # self.sanity_check(direction, int(len(self.foreign_vocab) * 0.05))
                    pickle.dump((self.foreign_to_source_translation_probs, self.rare_tokens), model_dump)
                    print("Storing foreign language %s to source language %s model after %d iterations" % (self.foreign_language, self.source_language, iter_count))
                model_dump.close()

            if (self.stop_condition(iter_count)):
                print("Storing model after %d iterations" % (iter_count))
                model_dump = open(self.OUTPUT_FILE, 'wb')
                if direction == self.SOURCE_TO_FOREIGN:
                    # print("Spot checking on 5 percent of source %s 's vocabulary before storing!" % (self.source_language))
                    # self.sanity_check(direction, int(len(self.source_vocab) * 0.05))
                    pickle.dump((self.source_to_foreign_translation_probs, self.rare_tokens), model_dump)
                    print("Storing source language %s to foreign language %s translation model after %d iterations" % (
                    self.source_language, self.foreign_language, iter_count))
                elif direction == self.FOREIGN_TO_SOURCE:
                    # print(
                    # "Spot checking on 5 percent of foreign language %s vocabulary before storing!" % (self.foreign_language))
                    # self.sanity_check(direction, int(len(self.foreign_vocab) * 0.05))
                    pickle.dump((self.foreign_to_source_translation_probs, self.rare_tokens), model_dump)
                    print("Storing foreign language %s to source language %s model after %d iterations" % (
                    self.foreign_language, self.source_language, iter_count))
                model_dump.close()
                break

        print("Memory usage stats")
        print("Foreign lang %s vocab length: %d"%(self.foreign_language, len(self.foreign_vocab)))
        print("Source lang %s vocab length: %d"%(self.source_language, len(self.source_vocab)))
        print("No of cross product entries required: ", len(self.foreign_vocab) * len(self.source_vocab))

        if direction == self.SOURCE_TO_FOREIGN:
            print(
                "Num of conditional probabilities actually stored: ", len(self.source_to_foreign_translation_probs.keys()))
            print("Num of source_totals actually stored: ", len(self.source_totals.keys()))
        elif direction == self.FOREIGN_TO_SOURCE:
            print(
                "Num of conditional probabilities actually stored: ",
                len(self.foreign_to_source_translation_probs.keys()))
            print("Num of foreign_totals actually stored: ", len(self.foreign_totals.keys()))
        tot_counts = 0
        for key in self.counts.keys():
            tot_counts += len(self.counts[key].keys())
        print("Num of counts actually stored: ", tot_counts)

        self.sanity_check(direction)
        if direction == self.FOREIGN_TO_SOURCE:
            return(self.foreign_to_source_translation_probs)
        elif direction == self.SOURCE_TO_FOREIGN:
            return(self.source_to_foreign_translation_probs)

    def sanity_check(self, direction, n_sample=None):
        if direction == self.SOURCE_TO_FOREIGN:
            source_vocab = self.source_vocab
            target_vocab = self.foreign_vocab
            translation_probs = self.source_to_foreign_translation_probs
        elif direction == self.FOREIGN_TO_SOURCE:
            source_vocab = self.foreign_vocab
            target_vocab = self.source_vocab
            translation_probs = self.foreign_to_source_translation_probs
        if n_sample is None:
            test_source_words = [source_word for source_word in
                                 source_vocab]  # should not further stem words in english vocab
            print("Performing sanity check on full vocabulary")
        else:
            test_source_words = [
                source_word for
                source_word in random.sample(source_vocab, n_sample)]
            # print("Spot checking for the following words ")
            # print(test_source_words)

        for source_word in test_source_words:
            max_prob_word = None
            max_prob = 0.0
            tot_conditional_prob = 0.0
            for target_word in target_vocab:
                if translation_probs.get((target_word, source_word), 0.0) != 0.0:
                    tot_conditional_prob += translation_probs.get((target_word, source_word), 0.0)
                    if translation_probs.get((target_word, source_word), 0.0) > max_prob:
                        max_prob = translation_probs.get((target_word, source_word), 0.0)
                        max_prob_word = target_word
            assert abs(tot_conditional_prob - 1.0) < 0.000000000001, 'Tot conditional probability != 1 !!!'
            #if n_sample is not None:
            #    print("Most likely word for source word ", source_word, " is the target word ", max_prob_word, " with translation probability ", max_prob)
        print("Sanity check passed!")




class DE_Compound(Model1):

    def __init__(self,parameter_file):#,compounds_file="data/compound.dict"):
        super(DE_Compound,self).__init__(parameter_file)
        self.compounds = pickle.load(open("data/compound.dict",'rb'))#compounds_file)

    def get_parallel_instance(self, corpus_line):
        [german, english] = corpus_line.strip().split(' ||| ')
        german_stemmed_sentence,english_stemmed_sentence = \
            ([(i,self.get_german_stem(w).lower())
              for (i,word) in [(j,(wt.strip("„") if "„" in wt else wt))
                               for (j,wort) in enumerate(german.split(' '))
                               for wt in ([wort] if wort=='-' else wort.split('-'))]
              for w in ([word] if word not in self.compounds
                        else self.compounds[word])],
             [(i,self.get_english_stem(word).lower())
              for (i,word) in [(j,wt) for (j,wort) in enumerate(english.split(' '))
                               for wt in ([wort] if wort=='-' else wort.split('-'))]])
        return ([(i,self.rare_token) if word in self.rare_tokens[0] else (i, word)
                 for (i,word) in german_stemmed_sentence],
                [(i,self.rare_token) if word in self.rare_tokens[1] else (i, word)
                 for (i,word) in english_stemmed_sentence])

    def get_alignment(self, german, english):
        """
        Returns model1 alignment for a DE/EN parallel sentence pair.
        For each german word, identifies the best english word (or NULL) to align to
        """
        english.append((len(english),self.null_val))
        alignment = []
        for (i, g_i) in german:
            best = -1
            bestscore = 0
            for (j, e_j) in english:
                val = self.get_prior()*self.get_translation_prob(g_i,e_j)
                if best==-1 or val>bestscore:
                    bestscore = val
                    best = j
            if best < len(english)-1:
                yield (i,best) # don't yield anything for NULL alignment


class EM_DE_Compound(DE_Compound,EM_model1):

    def __init__(self, input_file, output_file, n_iterations):

        self.MAX_ITERS = n_iterations
        self.INPUT_FILE = input_file
        self.OUTPUT_FILE = output_file

        self.compounds = pickle.load(open("data/compound.dict",'rb'))#compounds_file)
        print("Accumulating word counts")
        self.rare_tokens = (set(), set())
        self.rare_tokens = self._find_rares()
        print("Done identifying rare words")


    def get_parallel_instance(self, corpus_line):
        """
        Removes the DE_Compound word indices for easier EM estimation
        """
        (german,english) = \
            super(EM_DE_Compound,self).get_parallel_instance(corpus_line)
        return ([w for (i,w) in german],[w for (i,w) in english])