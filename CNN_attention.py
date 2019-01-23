import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torch.autograd import Variable
import numpy as np
import pickle
import bcolz
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch.cuda
from sklearn.feature_extraction.text import TfidfVectorizer

DIM = 300
quest_size = 30
batch_size = 32
filters = 128
epochs = 10
randVec = np.random.uniform(-1, 1, DIM)
strip_char = '?,[](){}.<>:;" '

#=======================Creating Glove vector dict=============================
vectors = bcolz.open('glove/6B.300.dat')[:]
words = pickle.load(open('glove/6B.300_words.pkl','rb'))
word2idx = pickle.load(open('glove/6B.300_idx.pkl','rb'))

glove = {w: vectors[word2idx[w]] for w in words}

#=============================Finding Accuracy=================================
def get_accuracy(model_name, data):
    test_input = np.array(data[:,3:5],dtype=np.str)
    test_label = np.array(data[:,-1],dtype=np.float)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    N = test_input.shape[0]
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    for i in range(N):
        q1 = stringToGlove(test_input[i][0],quest_size)
        q2 = stringToGlove(test_input[i][1],quest_size)
        q1 = torch.cuda.FloatTensor(q1).view(1,1,quest_size,DIM)
        q2 = torch.cuda.FloatTensor(q2).view(1,1,quest_size,DIM)
        out = model(q1,q2)
        label = torch.cuda.FloatTensor(np.array(test_label[i]))
        out = 1 if out > 0.5 else 0
        if out == label : 
            if out == 1: TP += 1
            else: TN += 1
        else :
            if out == 1: FP += 1
            else : FN += 1

    accuracy = 100*(TP + TN)/N
    precision = 100*TP/(TP + FP)
    recall = 100*TP/(TP + FN)
    F1_score = 2*recall*precision/(recall + precision)
    return accuracy, precision, recall, F1_score


#=========================Creating sentence Matrix=============================
def stringToGlove(str, size):
    sv = []
    str = analyze(str)
    n = len(str)
    for i in range(size):
        if i < n:
            try:
                sv.append(glove[str[i].strip(strip_char).lower()])
            except KeyError:
                sv.append(randVec)
        else:
            sv.append([0.0]*DIM)

    return np.array(sv)

#==============================The dataset class==============================
class quoraDataset(Dataset):
    def __init__(self,data):
        self.train_data = np.array(data[:, 3:5],dtype=np.str)
        self.train_labels = np.array(data[:, -1])
        self.len = data.shape[0]

    def __getitem__(self,index):
        s1 = stringToGlove(self.train_data[index][0], quest_size)
        s2 = stringToGlove(self.train_data[index][1], quest_size)

        return s1,s2, self.train_labels[index]

    def __len__(self):
        return self.len

#==============================The model======================================
class convText(nn.Module):
    def __init__(self):
        super(convText,self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(1,filters,(1,DIM),bias=False))
        self.conv.append(nn.Conv2d(1,filters,(3,DIM),bias=False))
        self.conv.append(nn.Conv2d(1,filters,(5,DIM),bias=False))
        self.conv.append(nn.Conv2d(1,filters,(7,DIM),bias=False))

        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(512,1)
        self.bn = nn.BatchNorm2d(filters)

    def forward(self,x1,x2):
        pools_q1 = []
        pools_q2 = []

        for i,layer in enumerate(self.conv):
            conv_q1 = F.relu(self.bn(layer(x1)))#remember to put batchnorm layer
            conv_q2 = F.relu(self.bn(layer(x2)))

            attention = make_attention(conv_q1, conv_q2)
            q1_pool_factor = attention.sum(1)
            q2_pool_factor = attention.sum(2)
            q1_att = att_pool(conv_q1, q1_pool_factor)
            q2_att = att_pool(conv_q2, q2_pool_factor)

            pools_q1.append(q1_att)
            pools_q2.append(q2_att)

        q1_out = torch.cat(pools_q1,1)
        q2_out = torch.cat(pools_q2,1)

        q1_out = self.fc1(q1_out)
        q2_out = self.fc1(q2_out)

        # out = torch.cat((q1_out,q2_out),1)
        # out = F.sigmoid(self.fc2(out)).squeeze()
        
        cos = nn.CosineSimilarity(dim=1)
        out = cos(q1_out,q2_out)
        return out

def make_attention(x1,x2):
    x1 = x1.permute(0,3,2,1)
    x2 = x2.permute(0,2,3,1)

    eps = torch.tensor(1e-6).cuda()
    one = torch.tensor(1.).cuda()
    euclidean = (torch.pow(x1 - x2, 2).sum(dim=3) + eps).sqrt()
    ans = (euclidean + one).reciprocal()
    return ans

def att_pool(matrix, weights):
    matrix = matrix.squeeze()
    weights = weights.unsqueeze(1)
    return torch.sum(matrix*weights,2)

#===============================Train-Test Split=================================
data = pd.read_csv('Dataset/questions.csv')
data = pd.DataFrame.as_matrix(data)

vectorizer = TfidfVectorizer()
quest_data = np.concatenate((data[:,3],data[:,4]))
quest_data = np.array(quest_data, np.str)
vectorizer.fit(quest_data)
analyze = vectorizer.build_analyzer()

test_data = []
train_data = []
dev_data = []
c1 = 0
c2 = 0
for d in data:
    if d[5] == 0:
        if c1 < 5000:
            dev_data.append(d)
            c1 += 1
        elif c1 < 10000 :
            test_data.append(d)
            c1 += 1
        else :
            train_data.append(d)
    else:
        if c2 < 5000:
            dev_data.append(d)
            c2 += 1
        elif c2 < 10000:
            test_data.append(d)
            c2 += 1
        else :
            train_data.append(d)

dev_data = np.array(dev_data)
test_data = np.array(test_data)
train_data = np.array(train_data)

#===============================Main Function=================================
model = convText()
model.cuda();
criterion = nn.MSELoss()
# criterion = nn.BCELoss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
# optimizer = optim.Adam(model.parameters(), lr = 0.0001)

quoraQuest = quoraDataset(train_data)
train_loader = DataLoader(dataset = quoraQuest, batch_size = batch_size, shuffle = True)

#==============================training loop===================================
# model.load_state_dict(torch.load("CNN_LR_model9.pt"))
# for epoch in range(epochs):
#     total_loss = 0
#     t = 1
#     for sample in train_loader:
#         ques1, ques2, label = sample
#         ques1 = ques1.view(-1,1,quest_size,DIM)
#         ques2 = ques2.view(-1,1,quest_size,DIM)

#         out = model(ques1.float().cuda(), ques2.float().cuda())        

#         loss = criterion(out, label.float().cuda())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         t+=1
#         if (t % 1000 == 0):
#             print( epoch, t, 100*total_loss/t)
#     torch.save(model.state_dict(), "CNN_LR_model" + str(epoch) + ".pt")
#     print(epoch, get_accuracy("CNN_LR_model" + str(epoch) + ".pt", dev_data))

#===============================testing loop===================================

for i in range(15):
    print(get_accuracy("CNN_model" + str(i) + ".pt", test_data))

