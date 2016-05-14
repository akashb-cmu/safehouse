import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-ip", "--ip_file", help="Input file with the original parallel data",
                        default="./data/all_parallel_tur_data.txt",
                        type=str)

arg_parser.add_argument("-trg_ouput", "--foreign_output_file", help="Monolingual output file for the target language",
                        default="./data/eng_only.txt",
                        type=str)

arg_parser.add_argument("-src_ouput", "--source_output_file", help="Monolingual output file for the source language",
                        default="./data/tur_only.txt",
                        type=str)

args = arg_parser.parse_args()
print("Args used for this run:")
print(args)

IP_FILE = args.ip_file
SRC_OUTPUT = args.source_output_file
TRG_OUTPUT = args.foreign_output_file

with open(SRC_OUTPUT, 'w') as src_file:
    with open(TRG_OUTPUT, 'w') as trg_file:
        with open(IP_FILE, 'r') as ip_file:
            for line in ip_file:
                line = line.strip(" \t\r\n")
                if len(line) > 0:
                    [foreign, source] = [sent.strip(" \t\r\n") for sent in line.split("|||")]
                    src_file.write(source + " ")
                    trg_file.write(foreign + " ")