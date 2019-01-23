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
batch_size = 32
filters = 128
epochs = 10
resnetFeat = 100
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
    model = convText()
    model.cuda();
    model.load_state_dict(torch.load(model_name))
    model.eval()

    N = test_input.shape[0]
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    for i in range(N):
        q1 = [test_input[i][0]]
        q2 = [test_input[i][1]]
        q1_out = model(q1)
        q2_out = model(q2)
        cos = nn.CosineSimilarity(dim=1)
        out = cos(q1_out,q2_out)
        label = torch.FloatTensor(np.array(test_label[i]))
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
def stringToGlove(str):
    sv = []
    str = analyze(str)
    n = len(str)
    for i in range(max(7,n)):
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
        s1 = self.train_data[index][0]
        s2 = self.train_data[index][1]

        return s1,s2, self.train_labels[index]

    def __len__(self):
        return self.len

#==============================The model======================================
class resnetBlock(nn.Module):
    def __init__(self, feat):
        super(resnetBlock,self).__init__()
        self.fc = nn.Linear(feat,feat)
        self.bn = nn.BatchNorm1d(feat)

    def forward(self,x):
        residual = x
        out = self.bn(self.fc(x))
        # out = self.bn(self.fc(out))
        out += residual
        return F.relu(out)

class convText(nn.Module):
    def __init__(self):
        super(convText,self).__init__()
        self.conv1g = nn.Conv2d(1,filters,(1,DIM),bias=False)
        self.conv3g = nn.Conv2d(1,filters,(3,DIM),bias=False)
        self.conv5g = nn.Conv2d(1,filters,(5,DIM),bias=False)
        self.conv7g = nn.Conv2d(1,filters,(7,DIM),bias=False)

        self.fc1 = nn.Linear(512,256)
        self.bn = nn.BatchNorm1d(256)
        self.resnet = resnetBlock(256)

    def forward(self,x):
        in_size = len(x)
        out = []
        for i,string in enumerate(x):
            gl_mx = torch.cuda.FloatTensor(stringToGlove(string))
            gl_mx = gl_mx.view(1,1,-1,DIM)

            c1g = torch.sum(F.relu(self.conv1g(gl_mx)),2)
            c3g = torch.sum(F.relu(self.conv3g(gl_mx)),2)
            c5g = torch.sum(F.relu(self.conv5g(gl_mx)),2)
            c7g = torch.sum(F.relu(self.conv7g(gl_mx)),2)

            c1g = c1g.view(1,-1)
            c3g = c3g.view(1,-1)
            c5g = c5g.view(1,-1)
            c7g = c7g.view(1,-1)

            out.append(torch.cat((c1g,c3g,c5g,c7g),1))

        out = torch.cat(out,0)
        out = self.fc1(out)
        
        return out

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
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
# optimizer = optim.Adam(model.parameters(), lr = 0.0001)

quoraQuest = quoraDataset(train_data)
train_loader = DataLoader(dataset = quoraQuest, batch_size = batch_size, shuffle = True)

#==============================training loop===================================
# model.load_state_dict(torch.load("CNN_model5.pt"))
# for epoch in range(epochs):
#     total_loss = 0
#     t = 1
#     for sample in train_loader:
#         ques1, ques2, label = sample

#         q1_out = model(ques1)
#         q2_out = model(ques2)
#         cos = nn.CosineSimilarity(dim=1)
#         out = cos(q1_out,q2_out)

#         loss = criterion(out, label.float().cuda())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         t+=1
#         # if (t % 1000 == 0):
#         #     print( epoch, t, 100*total_loss/t)
#     torch.save(model.state_dict(), "CNN_model" + str(epoch + 6) + ".pt")
#     print(epoch + 6, get_accuracy("CNN_model" + str(epoch + 6) + ".pt", dev_data))

#===============================testing loop===================================

for i in range(11):
	print(get_accuracy("CNN_model" + str(i) + ".pt", test_data))