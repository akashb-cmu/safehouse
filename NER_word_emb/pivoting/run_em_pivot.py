from Model1_EM import *
import pdb
import argparse


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-ip", "--input_file", help="Input File", default="./data/em_test.txt",
                        type=str)

arg_parser.add_argument("-src", "--src_lang", help="Source language (RHS of |||)",
                        choices=["english", "german", "turkish", "uzbek"],
                        default="english",
                        type=str)

arg_parser.add_argument("-dir", "--direction", help="Target | Source vs. Source | Target",
                        choices=["src_given_trg", "trg_given_src"],
                        default="trg_given_src",
                        type=str)

arg_parser.add_argument("-trg", "--foreign_lang", help="Foreign language (LHS of |||)",
                        choices=["english", "german", "turkish", "uzbek"],
                        default="german",
                        type=str)

arg_parser.add_argument("-iters", "--max_iters", help="Maximum iterations to run the alignment model", default=10,
                        type=int)
arg_parser.add_argument("-rthresh", "--rare_threshold", help="Threshold below which a word is replaced by a rare token", default=1,
                        type=int)
arg_parser.add_argument("-freq", "--store_freq", help="Frequency at which to store the model", default=1,
                        type=int)
arg_parser.add_argument("-mname", "--model_name", help="Name of the model", default="temp_model",
                        type=str)

args = arg_parser.parse_args()
print("Args used for this run:")
print(args)

INPUT_FILE = args.input_file
MODEL_OUTPUT_NAME = "./models/" + args.model_name
MAX_ITERS = args.max_iters
STORE_FREQ = args.store_freq
SOURCE_LANG = args.src_lang
FOREIGN_LANG = args.foreign_lang
RARE_THRESH = args.rare_threshold

if args.direction == "trg_given_src":
    DIRECTION = EM_model1.SOURCE_TO_FOREIGN
else:
    DIRECTION = EM_model1.FOREIGN_TO_SOURCE

IBM_model1 = EM_model1(INPUT_FILE, MODEL_OUTPUT_NAME, MAX_ITERS, rare_threshold=RARE_THRESH, source_language=SOURCE_LANG, foreign_language=FOREIGN_LANG)
# Assumes each line in corpus is of the form: <foreign lang sentence> ||| <source lang sentence>

IBM_model1.estimate_params(DIRECTION, STORE_FREQ)
# print("The params are:")
# print(IBM_model1.get_params(DIRECTION))

IBM_model1.sanity_check(DIRECTION)
#Model that decomposes
#
# Decomp_model = EM_DE_Compound(INPUT_FILE, MODEL_OUTPUT_NAME, MAX_ITERS)
# Decomp_model.estimate_params(DIRECTION, STORE_FREQ)
# #print(Decomp_model.rare_tokens)
# print("No. of rare tokens combined:")
# print("German:" + str(len(Decomp_model.rare_tokens[0])))
# print("English:" + str(len(Decomp_model.rare_tokens[1])))
#
# print("Sanity checking finally!!")
#
# Decomp_model.sanity_check(DIRECTION)