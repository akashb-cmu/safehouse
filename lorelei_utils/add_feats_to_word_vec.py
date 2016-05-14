from string import punctuation
from os import walk

ALL_VECS_PATH = "./vec_files"
OP_VECS_PATH = "./mod_vecs"

VEC_FILE = "./bolt_wiki_hausa_vec.txt"
MOD_VEC_FILE = "./mod_bolt_wiki_hausa_vec.txt"
MAX_N = 3
RARE_THRESHOLD = 150

def append_ortho_feats(token, feat_vect):
    #Detect capitalization
    if any(alphabet.isupper() for alphabet in token):
        feat_vect.append(1.0)
    else:
        feat_vect.append(0.0)

    #Detect numbers
    if any(alphabet.isdigit() for alphabet in token):
        feat_vect.append(1.0)
    else:
        feat_vect.append(0.0)

    puncts = [punct.decode('utf-8') for punct in punctuation]

    #Detect punctuation

    if any(alphabet in puncts for alphabet in token):
        feat_vect.append(1.0)
        # Detect if it is only a punct value
        if len(token) == 1:
            feat_vect.append(1.0)
        else:
            feat_vect.append(0.0)
    else:
        feat_vect.append(0.0)
        feat_vect.append(0.0) # Since it is not just a punct value

    #Detect hyphen?

    #Detect length
    feat_vect.append(len(token))

def append_k_gram_feats(n_gram_prefix_vocab, n_gram_suffix_vocab, token, feat_vect, MAX_N):
    token_n_grams_prefix = {}
    token_n_grams_suffix = {}
    for n in range(1, MAX_N + 1, 1):
        if token_n_grams_prefix.get(n, None) is None:
            token_n_grams_prefix[n] = set()
        if token_n_grams_suffix.get(n, None) is None:
            token_n_grams_suffix[n] = set()
        for alpha_index, alphabet in enumerate(token): #Checking pre-fixes
            if alpha_index + n - 1 < len(token):
                token_n_grams_prefix[n].add(token[alpha_index: alpha_index + n])
            else:
                break
        if len(token) - MAX_N < 0:
            rev_token = token
        else:
            rev_token = token[len(token) - MAX_N:]
        for alpha_index, alphabet in enumerate(rev_token):
            if alpha_index + n - 1 < len(rev_token):
                token_n_grams_suffix[n].add(rev_token[alpha_index: alpha_index + n])
            else:
                break

    for n in range(1, MAX_N + 1, 1):
        for n_gram in n_gram_prefix_vocab[n]:
            if n_gram in token_n_grams_prefix[n]:
                feat_vect.append(1.0)
            else:
                feat_vect.append(0.0) # Should this perhaps be -1.0
        for n_gram in n_gram_suffix_vocab[n]:
            if n_gram in token_n_grams_suffix[n]:
                feat_vect.append(1.0)
            else:
                feat_vect.append(0.0)


def append_feats(n_gram_prefix_vocab, n_gram_suffix_vocab, token, feat_vect, MAX_N):
    append_ortho_feats(token, feat_vect)
    append_k_gram_feats(n_gram_prefix_vocab, n_gram_suffix_vocab, token, feat_vect, MAX_N)

def get_upto_n_gram_vocab(VEC_FILE, MAX_N):
    global RARE_THRESHOLD
    n_gram_prefix_vocab = {}
    n_gram_suffix_vocab = {}
    with open(VEC_FILE, 'r') as ip_file:
        (vocab_size, dim) = ip_file.readline().split()
        for line in ip_file:
            [token, vec] = line.strip(' \t\r\n').split(' ', 1)
            token = token.decode('utf-8')
            #print(token)
            for n in range(1, MAX_N + 1):
                if n_gram_prefix_vocab.get(n, None) is None:
                    n_gram_prefix_vocab[n] = {}
                if n_gram_suffix_vocab.get(n, None) is None:
                    n_gram_suffix_vocab[n] = {}
                for alpha_index, alphabet in enumerate(token[0:MAX_N + 1]):
                    if alpha_index + n - 1 < len(token):
                        n_gram_prefix_vocab[n][token[alpha_index : alpha_index + n]] = n_gram_prefix_vocab[n].get(token[alpha_index : alpha_index + n], 0) + 1
                    else:
                        break
                if len(token) - MAX_N < 0:
                    rev_token = token
                else:
                    rev_token = token[len(token) - MAX_N:]
                for alpha_index, alphabet in enumerate(rev_token):
                    if alpha_index + n - 1 < len(rev_token):
                        n_gram_suffix_vocab[n][rev_token[alpha_index: alpha_index + n]] = n_gram_suffix_vocab[n].get(rev_token[alpha_index: alpha_index + n], 0) + 1
                    else:
                        break
    for n in range(1, MAX_N + 1):
        #n_prefix_vocab_list = list(n_gram_prefix_vocab[n])
        n_prefix_vocab_list = [prefix for prefix in n_gram_prefix_vocab[n].keys() if n_gram_prefix_vocab[n][prefix] > RARE_THRESHOLD]
        n_prefix_vocab_list.sort()
        n_gram_prefix_vocab[n] = n_prefix_vocab_list # enforcing ordering on features via list
        #n_suffix_vocab_list = list(n_gram_suffix_vocab[n])
        n_suffix_vocab_list = [suffix for suffix in n_gram_suffix_vocab[n].keys() if n_gram_suffix_vocab[n][suffix] > RARE_THRESHOLD]
        n_suffix_vocab_list.sort()
        n_gram_suffix_vocab[n] = n_suffix_vocab_list
    return([n_gram_prefix_vocab, n_gram_suffix_vocab])



for [path, dirs, files] in walk(ALL_VECS_PATH):
    for VEC_FILE in files:
        MOD_VEC_FILE = OP_VECS_PATH + "/mod_" + VEC_FILE
        VEC_FILE = path + "/" + VEC_FILE

        [n_gram_prefix_vocab, n_gram_suffix_vocab] = get_upto_n_gram_vocab(VEC_FILE, MAX_N)
        # print(n_gram_prefix_vocab)

        header_flag = 1
        print("Processing the file : " + VEC_FILE)
        with open(VEC_FILE, 'r') as ip_file:
            with open(MOD_VEC_FILE, 'w') as op_file:
                [vocab_size, vec_dim] = ip_file.readline().strip(' \t\r\n').split()
                print(vocab_size, vec_dim)
                for line in  ip_file:
                    [token_orig, vec] = line.strip(' \t\r\n').split(' ', 1) #encode back before writing
                    token = token_orig.decode('utf-8')
                    #print("Processing word " + token)
                    vec = [float(val) for val in vec.strip(' \t\r\n').split()]
                    #print("Original vec has length ", len(vec))
                    append_feats(n_gram_prefix_vocab, n_gram_suffix_vocab, token, vec, MAX_N)
                    #print("Modified vec has length ", len(vec))
                    print_vec = " ".join([str(val) for val in vec])
                    if header_flag == 1:
                        header_flag = 0
                        op_file.write(str(vocab_size) + " " + str(len(vec)) + "\n")
                    op_file.write(" ".join([token.encode('utf-8'), print_vec]) + "\n")