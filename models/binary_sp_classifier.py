import os
import pickle
import torch.nn.functional as F
import torch
import torch.nn as nn


def get_data_folder():
    if os.path.exists("/scratch/work/dumitra1"):
        return "/scratch/work/dumitra1/sp_data/"
    elif os.path.exists("/home/alex"):
        return "sp_data/"
    else:
        return "/scratch/project2003818/dumitra1/sp_data/"


class BinarySPClassifier(nn.Module):
    def __init__(self, input_size, output_size, classarray=False, filters=[120, 100, 80, 60], lengths=[5, 9, 15, 21],
                 dos=[0.1, 0.2]):
        super(BinarySPClassifier, self).__init__()
        self.extract_embeddings = False

        # filters=[120,100,80,60] # dimensionality of outputspace
        n_f = sum(filters)
        # lengths=[5,10,15,20] # original kernel sizes
        # lengths=[5,9,15,21] # kernel sizes (compatible with pytorch and 'same' padding used in TF)
        self.useSoftmax = ~classarray
        self.dos = dos

        self.output_size = output_size
        self.cnn1 = nn.Conv1d(input_size, filters[0], kernel_size=lengths[0], padding=(lengths[0] - 1) // 2)
        self.cnn2 = nn.Conv1d(input_size, filters[1], kernel_size=lengths[1], padding=(lengths[1] - 1) // 2)
        self.cnn3 = nn.Conv1d(input_size, filters[2], kernel_size=lengths[2], padding=(lengths[2] - 1) // 2)
        self.cnn4 = nn.Conv1d(input_size, filters[3], kernel_size=lengths[3], padding=(lengths[3] - 1) // 2)

        self.bn = nn.BatchNorm1d(n_f)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.cnn5 = nn.Conv1d(n_f, 100, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(100)
            # 27 * 1024,  128
        # 27 * 1024,  128
        # self.dense = nn.Linear(100,1)
        self.dense = nn.Linear(100, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, extract_embeddings=False):
        x = torch.cat((self.cnn1(x), self.cnn2(x), self.cnn3(x), self.cnn4(x)), dim=1)
        x = self.bn(x)
        x = self.relu(x)

        x = nn.functional.dropout(x, self.dos[0])

        x = self.cnn5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = torch.squeeze(x, dim=2)
        if extract_embeddings:
            return x
        x = nn.functional.dropout(x, self.dos[1])

        x = self.dense(x)
        return x

class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layers=1):
        super(ConvResBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        if layers == 2:
            self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size,padding=padding)
            self.bn2 = nn.BatchNorm1d(out_channels)
        self.layers = layers
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.layers == 1:

            x_ = self.bn1(self.conv(x))
            return self.relu(x + x_)
        else:
            x_ = self.relu(self.bn1(self.conv(x)))
            x_ = self.bn2(self.conv2(x_))
            return self.relu(x + x_)


class ResBlock(nn.Module):
    def __init__(self, dim, dos, no_layers=1):
        super(ResBlock, self).__init__()
        self.linear = nn.Linear(dim, dim)
        if no_layers == 2:
            self.linear2 = nn.Linear(dim, dim)
        # self.linear2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dos)
        self.layer_norm = nn.LayerNorm(dim)
        if no_layers == 2:
            self.layer_norm2 = nn.LayerNorm(dim)
        # self.layer_norm2 = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.no_layers = no_layers

    def forward(self, x):
        x_ = self.layer_norm(self.linear(x))
        if self.no_layers == 2:
            x_ = self.layer_norm2(self.linear2(self.relu(x_)))
        # x_ = self.layer_norm2(self.linear2(x_))
        return self.relu(x + x_)

class CNN3(nn.Module):
    def __init__(self,input_size,output_size,filters=[120,100,80,60],lengths=[5,9,15,21,3],dos=[0,0],pool='sum',is_cnn2=False,deep_mdl=False, no_of_layers=4, cnn_resnets=4,
                 add_additional_emb=True):
        super(CNN3, self).__init__()
        aa_dict = pickle.load(open("sp6_dicts.bin", "rb"))
        aa_dict = {k:v for k,v in aa_dict[-1].items() if v not in ['ES','PD','BS']}
        self.add_additional_emb = add_additional_emb
        self.deep_mdl=deep_mdl
        self.residue_emb = EmbModule(aa_dict)

        input_size = input_size + 128 if add_additional_emb else input_size
        #filters=[120,100,80,60] # dimensionality of outputspace
        n_f = sum(filters)
        #lengths=[5,10,15,20] # original kernel sizes
        #lengths=[5,9,15,21] # kernel sizes (compatible with pytorch and 'same' padding used in TF)
        self.dos=dos
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_cnn2 = is_cnn2

        self.output_size=output_size
        self.cnn1 = nn.Conv1d(input_size,filters[0],kernel_size=lengths[0],padding=(lengths[0]-1)//2)
        self.cnn2 = nn.Conv1d(input_size,filters[1],kernel_size=lengths[1],padding=(lengths[1]-1)//2)
        self.cnn3 = nn.Conv1d(input_size,filters[2],kernel_size=lengths[2],padding=(lengths[2]-1)//2)
        self.cnn4 = nn.Conv1d(input_size,filters[3],kernel_size=lengths[3],padding=(lengths[3]-1)//2)
        self.drop1 = nn.Dropout(dos[0])
        self.drop2 = nn.Dropout(dos[1])
        self.bn= nn.BatchNorm1d(n_f)
        self.relu = nn.ReLU()
        if pool=='sum':
            self.pool = nn.AdaptiveMaxPool1d(1)
        else: # pool=='avg'
            self.pool = nn.AdaptiveAvgPool1d(1)

        self.cnn5 = nn.Conv1d(n_f,100,kernel_size=lengths[4],padding=(lengths[4]-1)//2)
        self.bn5 = nn.BatchNorm1d(100)
        self.dense_i = nn.Linear(input_size,256)
        self.res_layers = []
        if cnn_resnets != 0:
            for i in range(cnn_resnets):
                self.res_layers.append(ConvResBlock(100, 100, kernel_size=5))
            self.conv_res_layers = nn.Sequential(*self.res_layers)
        else:
            self.conv_res_layers = None
        self.res_layers = []
        if self.deep_mdl:
            for i in range(no_of_layers):
                self.res_layers.append(ResBlock(100 if self.is_cnn2 else 100+256, dos[0]))
            self.res_layers = nn.Sequential(*self.res_layers)
            self.dense = nn.Linear(100,output_size) if is_cnn2 else nn.Linear(100+256,output_size)
        else:
            self.dense = nn.Linear(100,output_size) if is_cnn2 else nn.Linear(100+256,output_size)




        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self,x, targets=None, inp_seqs=None):
        #print('0. SHAPE x',x.shape)
        # CNN part
        x = x.permute(0,2,1)
        if self.add_additional_emb:
            additional_imp = self.residue_emb(inp_seqs)
            additional_imp = additional_imp.permute(1,2,0)
            x = torch.cat([additional_imp, x], dim=1)
        xc = torch.cat((self.cnn1(x),self.cnn2(x),self.cnn3(x),self.cnn4(x)),dim=1)
        xc = self.bn(xc)
        xc = self.relu(xc)
        xc = self.drop1(xc)
        xc = self.cnn5(xc)
        xc = self.bn5(xc)
        xc = self.relu(xc)
        if self.conv_res_layers is not None:
            xc = self.conv_res_layers(xc)

        #print('1. SHAPE x after convolutions',xc.shape)
        xc = self.pool(xc)
        xc = torch.squeeze(xc,dim=2)

        # PARALLEL LNN part
        if not self.is_cnn2:
            x = self.pool(x)
            x = torch.squeeze(x,dim=2)
            x = self.dense_i(x)
            x = self.relu(x)


            # COMBINED part
            x = torch.cat((xc,x),dim=1)
            #print('SHAPE x after cat (xc,x)',x.shape)

            x = self.drop2(x)
            #x = self.softmax(x)
            #return x
            if self.deep_mdl:
                x = self.res_layers(x)
            x = self.dense(x)
        else:
            if self.deep_mdl:
                x = self.res_layers(xc)
            else:
                x = xc
            x = self.dense(x)

        return x

class EmbModule(nn.Module):
    def __init__(self, aa_dict, emb_dim=128, og=False):
        super(EmbModule, self).__init__()
        og_dict = {'EUKARYA':0, 'NEGATIVE':1, 'POSITIVE':2, 'ARCHAEA':3}
        self.og = og
        if og:
            self.emb = nn.Embedding(len(og_dict.items()), emb_dim)
        else:
            self.emb = nn.Embedding(len(aa_dict.items()), emb_dim)

        self.aa_dict = aa_dict
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, seqs):
        emb_seqs = []
        max_s_l = max([len(s_) for s_ in seqs])
        for s_ in seqs:
            if self.og:
                og_emb = self.emb(self.aa_dict[s_].reshape(-1,1).repeat(1,max_s_l))
                emb_seqs.append(og_emb)
            else:
                s_ = s_ + (max_s_l - len(s_))*"X"
                input_indices = torch.tensor([self.aa_dict[c] for c in s_]).to(self.device)
                emb_seqs.append(self.emb(input_indices))
        return torch.nn.utils.rnn.pad_sequence(emb_seqs)


class CNN4(nn.Module):
    def __init__(self,input_size,output_size,filters=[120,100,80,60],lengths=[5,9,15,21,3],dos=[0,0],pool='avg',is_cnn2=False,deep_mdl=False, no_of_layers=4, cnn_resnets=4,
                 add_additional_emb=True, add_emb_dim=32, og_emb_dim=32):
        super(CNN4, self).__init__()
        self.form_lg_dict()
        aa_dict = pickle.load(open("sp6_dicts.bin", "rb"))
        aa_dict = {k:v for k,v in aa_dict[-1].items() if k not in ['ES','PD','BS']}
        aa_dict['X']= 20
        self.add_additional_emb = add_additional_emb
        input_size = input_size + add_emb_dim + og_emb_dim if add_additional_emb else input_size
        self.residue_emb = EmbModule(aa_dict, emb_dim=add_emb_dim)
        self.og_emb = EmbModule(aa_dict, emb_dim=og_emb_dim) if og_emb_dim != 0 else None
        pool = 'max'
        self.deep_mdl=deep_mdl
        if pool=='max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        else: # pool=='avg'
            self.pool = nn.AdaptiveAvgPool1d(1)
        #filters=[120,100,80,60] # dimensionality of outputspace
        n_f = sum(filters)
        #lengths=[5,10,15,20] # original kernel sizes
        #lengths=[5,9,15,21] # kernel sizes (compatible with pytorch and 'same' padding used in TF)
        self.dos=dos
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_cnn2 = is_cnn2
        self.relu = nn.ReLU()

        self.output_size=output_size
        self.cnn_reduce1 = nn.Conv1d(input_size,512,kernel_size=7,padding=3,stride=1)
        self.reduce_pool_1 = nn.MaxPool1d(3, 2)
        self.bn1 = nn.BatchNorm1d(512)
        self.cnn_reduce2 = nn.Conv1d(512,256,kernel_size=7,padding=3, stride=1)
        self.reduce_pool_2 = nn.MaxPool1d(3, 2)
        self.bn2 = nn.BatchNorm1d(256)

        res_layers = []
        if cnn_resnets != 0:
            for i in range(cnn_resnets):
                res_layers.append(ConvResBlock(256, 256, kernel_size=5))
            self.conv_res_layers1 = nn.Sequential(*res_layers)
        self.cnn_reduce3 = nn.Conv1d(256,256,kernel_size=5,padding=2, stride=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.reduce_pool_3 = nn.MaxPool1d(3, 2)

        res_layers = []
        if cnn_resnets != 0:
            for i in range(cnn_resnets):
                res_layers.append(ConvResBlock(256, 256, kernel_size=3))
            self.conv_res_layers2 = nn.Sequential(*res_layers)

        self.dense = nn.Linear(256,output_size) if is_cnn2 else nn.Linear(256,output_size)

    def forward(self,x, targets=None, inp_seqs=None):
        #print('0. SHAPE x',x.shape)
        # CNN part
        x = x.permute(0,2,1)
        if self.add_additional_emb:
            additional_imp = self.residue_emb(inp_seqs)
            additional_imp = additional_imp.permute(1,2,0)
            x = torch.cat([additional_imp, x], dim=1)
        if self.og_emb is not None:
            additional_imp = self.og_emb(inp_seqs)
            additional_imp = additional_imp.permute(1, 2, 0)
            x = torch.cat([additional_imp, x], dim=1)
        x = self.reduce_pool_1(self.relu(self.bn1(self.cnn_reduce1(x))))
        x = self.reduce_pool_2(self.relu(self.bn2(self.cnn_reduce2(x))))
        x = self.conv_res_layers1(x)
        x = self.reduce_pool_3(self.relu(self.bn3(self.cnn_reduce3(x))))
        x = self.conv_res_layers2(x)
        x = self.pool(x)
        x = torch.squeeze(x,dim=2)
        return self.dense(x)

    def form_lg_dict(self):
        folder = get_data_folder()
        seq2og = {}
        for tr_f in [0,1,2]:
            for t in ['train','test']:
                data = pickle.load(open(folder+"sp6_partitioned_data_sublbls_{}_{}.bin".format(t,tr_f),"rb"))
                current_seq2og = {seq: v[-2] for seq, v in data.items()}
                seq2og.update(current_seq2og)
        self.seq2og = seq2og



