import pandas as pd

def extract_fasta():
    rest_of_seq = "TRATADAQSRMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPP" \
                  "DQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGVVGKLGRQDPPVATR"
    df = pd.read_csv("N_sequences.txt", sep='\t')
    names, seq = df.ID, df.AA_seq
    index = 0
    for ind, (n, s) in enumerate(zip(names, seq)):
        if ind % 5000 == 0:
            f = open("sp_seq_data_file_{}.fasta".format(index), "wt")
            index += 1
        f.write(">" + n + "\n")
        f.write(s + rest_of_seq[:70-len(s)]+ "\n")

if __name__ == "__main__":
    extract_fasta()