from Bio import SeqIO

sp1_methods = {"sp6", "sp5", "DEEPSIG", "LipoP", "PHILIUS", "PHOBIUS", "PolyPhobius", "PRED-LIPO", "PRED-SIGNAL", "PRED-TAT", "SIGNAL-CF", "Signal-3L", "SOSUIsignal", "SPEPlip", "SPOCTOPUS", "TOPCONS2", }
sp2_methods = {"LipoP", "PRED-LIPO", "SPEPlip", }
tat_methods = {"PRED-TAT", "TatP", "TATFIND",}
print(sp1_methods.intersection(sp2_methods))
f1 = "sp_data/sp6_data/train_set.fasta"
f2 = "sp_data/sp6_data/benchmark_set_sp5.fasta"
seqs, lbls, ids, uid = [], [], [], []
spandlg2count = {}
for seq_record in SeqIO.parse(f1, "fasta"):
    seqs.append(str(seq_record.seq[:len(seq_record.seq) // 2]))
    lbls.append(seq_record.seq[len(seq_record.seq) // 2:])
    ids.append(seq_record.id)
    uid.append(seq_record.id.split("|")[-1])
    if "_".join(seq_record.id.split("|")[1:3]) in spandlg2count:
        spandlg2count["_".join(seq_record.id.split("|")[1:3])] += 1
    else:
        spandlg2count["_".join(seq_record.id.split("|")[1:3])] = 1
seqs2, lbls2, ids2, uid2 = [], [], [], []
print(spandlg2count)
print(set([id_.split("|")[-1] for id_ in ids]))
for seq_record in SeqIO.parse(f2, "fasta"):
    seqs2.append(str(seq_record.seq[:len(seq_record.seq) // 2]))
    # lbls.append(seq_record.seq[len(seq_record.seq) // 2:])
    # ids.append(seq_record.id)
    # uid.append(seq_record.id.split("|")[-1])

print(len(set(seqs2).intersection(set(seqs))), len(seqs2), len(seqs))
