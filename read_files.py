from Bio import SeqIO
ind = 0
unique_seq_lbls = set()
unique_types = set()
unique_kingdoms = set()
unique_partitionso = set()
def check_peptide(str, type='lipo'):
    ind = 0
    sp_aa = 'L' if type=='lipo' else 'T'
    while str[ind] == sp_aa:
        ind += 1
    while ind < len(str):
        if str[ind] != "O":
            return True
        ind +=1
    return False
a_ = 0
for seq_record in SeqIO.parse("train_set.fasta", "fasta"):
    a_ += 1
    # print(len(seq_record.seq)/2)
    unique_seq_lbls.update([s_ for s_ in str(seq_record.seq[int(len(seq_record.seq)/2):])])
    unique_kingdoms.add(seq_record.id.split("|")[1])
    unique_partitionso.add(seq_record.id.split("|")[-1])
    unique_types.add(seq_record.id.split("|")[2])
    if seq_record.id.split("|")[2] == "LIPO":
        # if check_lipo(seq_record.seq[int(len(seq_record.seq) / 2):]):
        if check_peptide(seq_record.seq[int(len(seq_record.seq) / 2):], type='tat'):
            print(seq_record.id)
            print(seq_record.seq)
    # if len(seq_record.seq) != 140:
    #     ind += 1
    #     print(seq_record.id)
    #     print(seq_record.seq)
        # print("YO", len(seq_record.seq))
    # print(len(seq_record))
print(unique_seq_lbls, unique_kingdoms,unique_partitionso,unique_types)
print(a_)