import pickle
import pandas as pd
def map_protein_to_sps(funct_sps):
    lines = []
    with open("prot_sps_bs5.txt", "rt") as f:
        lines = f.readlines()
    current_protein = ""
    sp2protein = {}
    sp_prot_pairs = []
    for l in lines:
        current_protein = l.replace("$", "").replace(".","") if "$" in l else current_protein
        if "(75)" in l or "(90)" in l or "(95)" in l or "(99)" in l:
            sp2protein[l.split(".")[0]] = current_protein
    funct_sps = [fs[1] for fs in funct_sps]
    for fs in funct_sps:
        sp_prot_pairs.append((fs, sp2protein[fs].replace("\n","")))
    return sp_prot_pairs

def create_70_aa_seq_files(sp_prt_pairs):
    # generate binary file for TSignal
    seqs, embs, lbls, glbl_lbls = [], [], [], []
    for sp, prot in sp_prt_pairs:
        seqs.append(sp + prot[:70-len(sp)])
        embs.append([1])
        lbls.append("S"*len(sp) + "O"* len(prot))
        glbl_lbls.append("SP")
    seq2info = {}
    for s,e,l,gl in zip(seqs, embs, lbls, glbl_lbls):
        seq2info[s] = [e, l, 'POSITIVE', gl]
    pickle.dump(seq2info, open("synthetic_gp_prot_sp_pairs.bin", "wb"))
    # generate fasta file for SignalP 6.0
    with open("synthetic_gp_bacteria_sps.fasta", "wt") as f:
        for ind, s in enumerate(seqs):
            f.write(">seq{}\n".format(ind))
            f.write(s+"\n")

functional_sps = pd.read_excel("Functional_generated_SPs_from_Wu_et_al.(2020).xlsx")
sp_prot_pairs = map_protein_to_sps(functional_sps.values)
create_70_aa_seq_files(sp_prot_pairs)