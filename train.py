import nltk
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader

nltk.download('punkt_tab')
lines=open("/Users/srikrishnaadithyakatragadda/Downloads/commentry_lines.rtf",encoding="utf-8").readlines()
lines=lines[9:]
total_lines=""
for i in range(len(lines)):
    line=lines[i]
    lines[i]=line[0:len(line)-3].lower()
    print(lines[i])
    total_lines=total_lines+lines[i]+" "

#Creating word to index mapping
tokenized=word_tokenize(total_lines) #All the lines are seperated into tokens
tokenized=set(tokenized)#Contains all the unique tokens needed for vocabulary
vocab={'<unk>':0}
for word in tokenized:
    vocab[word]=len(vocab)



#Preprocessing the data to get into required form
#Converting it so that we have a required supervised learning with an input and output
input=[]
output=[]

for line in lines:
    arr=word_tokenize(line.lower())
    inp=[]
    for i in range(len(arr)-1):
        out=[]
        inp.append(vocab[arr[i]])
        out.append(vocab[arr[i+1]])
        input.append(inp.copy())
        output.append(out.copy())
        out=[]

print(input)
print(len(output))

#Creating dataset and dataloader to fetch data from input and output

class dataset(Dataset):
    def __init__(self,input,output,vocab):
        self.input=input
        self.output=output
        self.vocab=vocab
    def __len__(self):
        return len(self.input)
    def __getitem__(self,index):
        x=self.input[index]
        y=self.output[index]
        #Padding at the end
        x=[0]*(64-len(x))+x
        return torch.tensor(x),torch.tensor(y)

train_dataloader=DataLoader(dataset(input,output,vocab),batch_size=64,shuffle=True)
test_dataloader=DataLoader(dataset(input,output,vocab),batch_size=64,shuffle=True)

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(len(vocab),50)
        self.lstm=nn.LSTM(50,150,batch_first=True)
        self.linear=nn.Linear(150,len(vocab))
    def forward(self,x):
        x=self.embedding(x)
        _,(hidden,_)=self.lstm(x)
        res=self.linear(hidden[0])
        return res

model=model()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

if __name__ == "__main__":
    for epoch in range(150):
        total_loss=0
        for x,y in train_dataloader:

            a=model(x)
            y=y.squeeze()
            loss=criterion(a,y)
            total_loss=total_loss+loss.item()
            optimizer.zero_grad()
            loss.backward()#We do for the present ones optimizer.zero_grad
            optimizer.step()
        print(f"Epoch {epoch} loss: {total_loss}")
        torch.save(model.state_dict(),"model_weight.pth")
















