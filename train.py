import nltk
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader

nltk.download('punkt_tab', quiet=True)

lines = open("/Users/srikrishnaadithyakatragadda/Downloads/commentry_lines.rtf", encoding="utf-8").readlines()
lines = lines[9:]
total_lines = ""
for i in range(len(lines)):
    line = lines[i]
    lines[i] = line[0:len(line) - 3].lower()
    total_lines = total_lines + lines[i] + " "

tokenized = sorted(list(set(word_tokenize(total_lines))))
vocab = {'<PAD>': 0}
for word in tokenized:
    if word not in vocab:
        vocab[word] = len(vocab)

#2. MODEL DEFINITION

class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 50, padding_idx=0)
        self.lstm = nn.LSTM(50, 150, batch_first=True)
        self.linear = nn.Linear(150, vocab_size)

    def forward(self, x, seq_lengths):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        # Pluck the exact output vector for the LAST REAL WORD
        batch_size = x.size(0)
        last_word_indices = seq_lengths - 1
        last_outputs = out[torch.arange(batch_size), last_word_indices]
        res = self.linear(last_outputs)
        return res


# 3. TRAINING LOOP
if __name__ == "__main__":

    input_seqs = []
    output_labels = []

    for line in lines:
        arr = word_tokenize(line.lower())
        for i in range(1, len(arr)):
            inp = [vocab[w] for w in arr[:i]]
            out = vocab[arr[i]]
            input_seqs.append(inp)
            output_labels.append(out)

    MAX_LEN = 20

    class CricketDataset(Dataset):
        def __init__(self, inputs, outputs):
            self.inputs = inputs
            self.outputs = outputs

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, index):
            x = self.inputs[index]
            y = self.outputs[index]
            seq_length=len(x)
            x_padded = x + [0] * (MAX_LEN - len(x))
            return torch.tensor(x_padded), torch.tensor(y), seq_length


    train_dataloader = DataLoader(CricketDataset(input_seqs, output_labels), batch_size=16, shuffle=True)

    model = NextWordLSTM(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print("Starting training...")
    for epoch in range(50):
        total_loss = 0
        for x_padded, y, seq_lengths in train_dataloader:
            optimizer.zero_grad()
            predictions = model(x_padded, seq_lengths)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model_weight.pth")
    print("Training complete and model saved!")