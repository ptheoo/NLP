# pos_rnn_ud.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# =======================
# Task 1: Load .conllu data
# =======================
def load_conllu(file_path):
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            word, upos = parts[1], parts[3]
            sentence.append((word, upos))
        if sentence:
            sentences.append(sentence)
    return sentences

# =======================
# Task 1b: Build vocabulary
# =======================
def build_vocab(sentences):
    word_to_ix = {"<UNK>":0}
    tag_to_ix = {}
    for sent in sentences:
        for word, tag in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    print(f"Vocabulary size: {len(word_to_ix)}")
    print(f"Number of tags: {len(tag_to_ix)}")
    return word_to_ix, tag_to_ix

# =======================
# Task 2: Dataset & DataLoader
# =======================
class POSDataset(Dataset):
    def __init__(self, sentences, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sent = self.sentences[idx]
        words_idx = [self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w, t in sent]
        tags_idx  = [self.tag_to_ix[t] for w, t in sent]
        return torch.tensor(words_idx, dtype=torch.long), torch.tensor(tags_idx, dtype=torch.long)

def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-100)  # ignore_index=-100
    return sentences_padded, tags_padded

# =======================
# Task 3: RNN model
# =======================
class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_tags)
    
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        logits = self.fc(out)  # batch x seq_len x num_tags
        return logits

# =======================
# Task 5: Evaluation
# =======================
def evaluate(model, loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            mask = y != -100
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    return correct / total if total > 0 else 0

def predict_sentence(model, sentence, word_to_ix, tag_to_ix, ix_to_tag, device='cpu'):
    model.eval()
    words_idx = [word_to_ix.get(w, word_to_ix["<UNK>"]) for w in sentence.split()]
    x = torch.tensor(words_idx, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
    return list(zip(sentence.split(), [ix_to_tag[p] for p in preds]))

# =======================
# Task 4: Training
# =======================
def train_model(model, train_loader, dev_loader, num_epochs=5, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    model.to(device)

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        train_acc = evaluate(model, train_loader, device)
        dev_acc = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Dev Acc={dev_acc:.4f}")
    
    print("Training finished.")
    return model

# =======================
# Main
# =======================
def main():
    # Tự động tìm thư mục data
    this_file = os.path.abspath(__file__)
    part3_dir = os.path.dirname(this_file)
    lab5_rnn_dir = os.path.dirname(part3_dir)
    project_root = os.path.dirname(lab5_rnn_dir)  # NLP
    data_dir = os.path.join(project_root, "data", "UD_English-EWT")

    train_file = os.path.join(data_dir, "en_ewt-ud-train.conllu")
    dev_file   = os.path.join(data_dir, "en_ewt-ud-dev.conllu")

    if not os.path.exists(train_file) or not os.path.exists(dev_file):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu trong {data_dir}")

    # Load dữ liệu
    train_sentences = load_conllu(train_file)
    dev_sentences = load_conllu(dev_file)

    # Build vocab
    word_to_ix, tag_to_ix = build_vocab(train_sentences)
    ix_to_tag = {v:k for k,v in tag_to_ix.items()}

    # Dataset & DataLoader
    train_dataset = POSDataset(train_sentences, word_to_ix, tag_to_ix)
    dev_dataset   = POSDataset(dev_sentences, word_to_ix, tag_to_ix)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Khởi tạo mô hình
    vocab_size = len(word_to_ix)
    num_tags = len(tag_to_ix)
    model = SimpleRNNForTokenClassification(vocab_size, embed_dim=100, hidden_dim=128, num_tags=num_tags)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_model(model, train_loader, dev_loader, num_epochs=5, device=device)

    # Test predict
    sentence = "The quick brown fox jumps over the lazy dog"
    print(predict_sentence(model, sentence, word_to_ix, tag_to_ix, ix_to_tag, device=device))

if __name__ == "__main__":
    main()
