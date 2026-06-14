import torch
from nltk.tokenize import word_tokenize

from train import NextWordLSTM, vocab

model = NextWordLSTM(len(vocab))
model.load_state_dict(torch.load("model_weight.pth"))
model.eval()

x_word = input("Enter your first word: ").strip().lower()
y_word = input("Enter your second word: ").strip().lower()
generated_words = [x_word, y_word]

for _ in range(10):
    encoded_seq = [vocab.get(w, 0) for w in generated_words]

    input_tensor = torch.tensor([encoded_seq])
    seq_length = torch.tensor([len(encoded_seq)])

    with torch.no_grad():
        prediction = model(input_tensor, seq_length)
        next_word_idx = torch.argmax(prediction, dim=1).item()

    next_word = "<unk>"
    for word, index in vocab.items():
        if index == next_word_idx:
            next_word = word
            break

    generated_words.append(next_word)

print("\nFinal Output:", " ".join(generated_words))
print("Testing done totally!!! ")