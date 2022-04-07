import torch.nn.functional as F
import torch
import torch.nn as nn

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


class CNN3(nn.Module):
    def __init__(self,input_size,output_size,filters=[120,100,80,60],lengths=[5,9,15,21,3],dos=[0.1,0.2],pool='sum'):
        super(CNN3, self).__init__()

        #filters=[120,100,80,60] # dimensionality of outputspace
        n_f = sum(filters)
        #lengths=[5,10,15,20] # original kernel sizes
        #lengths=[5,9,15,21] # kernel sizes (compatible with pytorch and 'same' padding used in TF)
        self.dos=dos

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
        self.dense = nn.Linear(100+256,output_size)

        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self,x, targets=None, inp_seqs=None):
        #print('0. SHAPE x',x.shape)
        # CNN part
        x = x.permute(0,2,1)
        xc = torch.cat((self.cnn1(x),self.cnn2(x),self.cnn3(x),self.cnn4(x)),dim=1)
        xc = self.bn(xc)
        xc = self.relu(xc)
        xc = self.drop1(xc)
        xc = self.cnn5(xc)
        xc = self.bn5(xc)
        xc = self.relu(xc)
        #print('1. SHAPE x after convolutions',xc.shape)
        xc = self.pool(xc)
        xc = torch.squeeze(xc,dim=2)

        # PARALLEL LNN part
        x = self.pool(x)
        x = torch.squeeze(x,dim=2)
        x = self.dense_i(x)
        x = self.relu(x)


        # COMBINED part
        x = torch.cat((xc,x),dim=1)
        #print('SHAPE x after cat (xc,x)',x.shape)

        x = self.drop2(x)
        x = self.dense(x)
        #x = self.softmax(x)
        #return x

        return x
