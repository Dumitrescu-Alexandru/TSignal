import numpy as np
from Bio import SeqIO
import pickle

def create_test_file(filename):
    # these lsit below contains actual sequences from  SignalP 6.0 dataset. Please process your files with e.g. biopython 
    # for fasta files, pickle for binaries, pandas for CSV/excel files; After you find a suitable way to extract the 
    # sequences, retrieve the first 70 residues of each sequence, and add all sequences to a list like the one below
    sequences = ['MNQDKSKETSELDDLALPRSIIMRLVKGVLPEKSLVQKEALKAMINSATLFVSFLTSASGEIATNNNRKI',
                 'MGLKGYSVGEGGGEIVEVQGGHIIRATGRKDRHSKVFTSKGPRDRRVRLSAHTAIQFYDVQDRLGYDRPS',
                 'MKPRKIILMGLRRSGKSSIQKVVFYKMPPNETLFLESTSKLTQDHISSFIDFSVWDFPGQVDVFDAAFDF',
                 'MGTPSHELNTTSSGAEVIQKTLEEGLGRRICVAQPVPFVPQVLGVMIGAGVAVLVTAVLILLVVRRLRVQ',
                 'MGDGLDAVQMSGSSSSQGQPSSQAPSSFNPNPPETSNPTRPKRQTNQLQYLLKVVLKSLWKHQFAWPFHA']
    # next, we use the same dataset loader for trainin and testing, and therefore the formats are kept the same:
    # data_dictionary of {sequence : (embedding, residue_labels, sp_type, organism_group). You can either put the true
    # information here, or have dummy entries (in this example, we have some dummy entries for each sequence added)
    # The predata_dictions will only be based on the actual sequences when using
    # e.g. python main.py --test_seqs <test_file_name> --test_mdl <file_name_mdl> ...; so add the correct info to the file
    # only if you want to use it later in some way

    max_len = 70
    data_dict = {}
    for s in sequences:
        data_dict[s[:max_len]] = [np.array([1]), len(s[:max_len]) * "I", "EUKARYA", "SP"]
    pickle.dump(data_dict, open(filename, "wb"))

def create_mammal_sequences():
        
    reporter_protein = "TRATADAQSRMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGVVGKLGRQDPPVATR"
    seqs, lbls, ids = [], [], []
    max_len = 70
    for ind, seq_record in enumerate(SeqIO.parse("hard_mammal_seqs.fasta", "fasta")):
        seq_ = str(seq_record.seq)
        seq_ += reporter_protein[:max_len-len(seq_)]
        seqs.append(seq_)
        ids.append("ID{}|EUKARYA|SP|0".format(ind))
        lbl_ = "S" * len(seq_record.seq)
        lbl_ += (max_len - len(lbl_)) * "I"
        lbls.append(lbl_)

    data_dict = {}
    for s, l, i in zip(seqs,lbls,ids):
        data_dict[s] = [np.array([1]), l, "EUKARYA", "SP"]

    pickle.dump(data_dict, open("test_seqs.bin", "wb"))

if __name__ == "__main__":

    # CREATE TEST FILE FOR THE HARD MAMMAL SEQUENCES (novel SPs)
    create_mammal_sequences()

    # EXAMPLE OF HOW TO PROCESS SOME SEQUENCE YOU WANT TO PREDICT
    create_test_file(filename="some_test_seqs.bin")