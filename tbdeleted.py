import matplotlib.pyplot as plt
a = [0.8074965219240372,0.8865215073007242,0.9091850452264456,0.9088374289196987,0.894618684406757,0.9105673610440674,0.9071865072447365,0.9029764980218454,0.9034963665307598,0.9036840683049597,
0.9054497794747358,0.9023446857138531,0.8998382949547479,0.9022186727543066,0.9169891941418591,0.8986603593483089,0.9030613672072456,0.9092818863431541,0.8823337401694279,0.8932541136879946,
0.9068168115117631,0.9172719838403369,0.9063772401260725,0.9191942169890764,0.9307160430435614,0.9258975524148231,0.9211883374541404,0.917346986158996,0.8867831203130588,0.9193845920120217,
0.9072606254109385,0.9020072417625522,0.9103189381292363,0.9065674714097618,0.8910398489237374,0.9195794281449907,0.939356743127283,0.8809910828970057,0.9008591278399948,0.9101360741992974,
0.9168240691887748,0.9492235753079931,0.9236078362869672,0.9177806951543195,0.9192153045105886,0.9186676528817626,0.9045496656658949,0.9217293379879238,0.9174665921715941,0.933878653054991,
     0.9151773390242604,0.9169528497469548,0.9002444133116327,0.9078267920776469,0.9061190494205726]


lines = open("bert_tuning_crct_swa_1_2.log", "rt")
mccs = []
for l in lines.readlines():
     if  " model with avg mcc " in l:
          mccs.append(float(l.split(" model with avg mcc ")[1].split(" ")[0]))
     elif "mcc was worse " in l:
          mccs.append(float(l.split("mcc was worse ")[1].split(" ")[0]))
lines = open("bert_tuning_deep_1_2.log", "rt")
mccs2 = []
for l in lines.readlines():
     if  " model with avg mcc " in l:
          mccs2.append(float(l.split(" model with avg mcc ")[1].split(" ")[0]))
     elif "mcc was worse " in l:
          mccs2.append(float(l.split("mcc was worse ")[1].split(" ")[0]))
plt.plot(a, label=' folds 1,2 2-layered;')
plt.plot(mccs,label='folds 1,2 2-layered; using lg')
plt.plot(mccs2,label='folds 1,2 4-layered; tuning bert from ep3')
plt.grid(axis='y', alpha=0.4, linewidth=0.4, color='black')
plt.legend()

plt.show()


a=[0.8686868185622764,0.9222818848428194,0.9273349355389913,0.9310411425375623,0.9009137081372408,0.8938915450204619,0.8936201397421668,0.9460387802615672,0.9030650670364153,0.9206661033996963,
0.967766950082884,0.9327411744452943,0.9195451622341778,0.9618813200513907,0.9541918715844597,0.9470044750356915,0.9271415899489677,0.9188764799070032,0.9623322124635624,0.9277785554554115,0.9292661149941621,
0.9036187367670669,0.9598679096350592,0.9535041957480834,0.9427354153043206,0.9554771647752,0.9359290245688293,0.9144679975840698,0.9216857500192717,0.9232296705350138,0.9265137211704101]
lines = open("bert_tuning_crct_swa_0_1.log", "rt")
mccs = []
for l in lines.readlines():
     if  " model with avg mcc " in l:
          mccs.append(float(l.split(" model with avg mcc ")[1].split(" ")[0]))
     elif "mcc was worse " in l:
          mccs.append(float(l.split("mcc was worse ")[1].split(" ")[0]))
lines = open("bert_tuning_deep_0_1.log", "rt")
mccs2 = []
for l in lines.readlines():
     if  " model with avg mcc " in l:
          mccs2.append(float(l.split(" model with avg mcc ")[1].split(" ")[0]))
     elif "mcc was worse " in l:
          mccs2.append(float(l.split("mcc was worse ")[1].split(" ")[0]))
plt.plot(a, label=' folds 1,1 2-layered;')
plt.plot(mccs,label='folds 0,1 2-layered; using lg')
plt.plot(mccs2,label='folds 0,1 4-layered; tuning bert from ep3')
plt.grid(axis='y', alpha=0.4, linewidth=0.4, color='black')
plt.legend()

plt.show()

a= [0.8879475020949523,0.8909592455732893,0.9038434554520405,0.9130774351304131,0.9334877061846996,0.9136280455055437,0.9136280455055437,0.9204984618616351,0.9358242131873811,0.910838709317534,
0.9304697662252658,0.9135191645180284,0.9231929527443856,0.9400125137993827,0.9388890941882762,0.9201346615023284,0.9415805338659821,0.923547529740061,0.9076940336062437,0.9518994283581458,
0.9307582263175987,0.9307582263175987,0.9370747371926896,0.9472047632661206,0.9213671664643711,0.9364913539578361,0.9444875647590354,0.9211754263793207,0.9310547878017443,0.9313067900396379,
0.9438269174266111,0.9208787417454017,0.9397274675564053,0.9191348731625566,0.929796786713773,0.9452384727047314,0.9379148793047669,0.9344717918477033,0.9418606268652221,0.9415805338659821,
0.9413760927531961]
lines = open("bert_tuning_crct_swa_0_2.log", "rt")
mccs = []
for l in lines.readlines():
     if  " model with avg mcc " in l:
          mccs.append(float(l.split(" model with avg mcc ")[1].split(" ")[0]))
     elif "mcc was worse " in l:
          mccs.append(float(l.split("mcc was worse ")[1].split(" ")[0]))
lines = open("bert_tuning_deep_0_2.log", "rt")
mccs2 = []
for l in lines.readlines():
     if  " model with avg mcc " in l:
          mccs2.append(float(l.split(" model with avg mcc ")[1].split(" ")[0]))
     elif "mcc was worse " in l:
          mccs2.append(float(l.split("mcc was worse ")[1].split(" ")[0]))
plt.plot(a, label=' folds 0,2 2-layered;')
plt.plot(mccs, label=' folds 0,2 2-layered; using lg')
plt.plot(mccs2,label='folds 0,2 4-layered; tuning bert from ep3')

plt.grid(axis='y', alpha=0.4, linewidth=0.4, color='black')
plt.legend()
plt.show()