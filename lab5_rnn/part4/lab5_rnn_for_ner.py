import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from seqeval.metrics import classification_report 
import time

# Đặt seed để đảm bảo tính khả lập lại
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Cấu hình Tham số ---
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.005
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_LABEL_ID = -100 

# Xác định thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Task 1: Tải và Tiền xử lý Dữ liệu ---

print("--- Task 1: Tải và Tiền xử lý Dữ liệu ---")

# 1. Tải dữ liệu từ Hugging Face
print("1. Tải dữ liệu CoNLL 2003...")
# BỎ trust_remote_code=True theo khuyến nghị của thư viện
dataset = load_dataset("conll2003") 
print("Tải dữ liệu hoàn tất.")

# 2. Trích xuất câu và nhãn, chuyển nhãn số sang string
tag_names = dataset["train"].features["ner_tags"].feature.names
print(f"Các nhãn NER: {tag_names}")

# Xây dựng từ điển nhãn ban đầu (Tag names -> Index)
tag_to_ix = {tag: i for i, tag in enumerate(tag_names)}
ix_to_tag = {i: tag for tag, i in tag_to_ix.items()}

# HÀM ÁNH XẠ NHÃN: Đảm bảo nhãn luôn là chuỗi trước khi trích xuất
def convert_to_string_tags_safe(example, tag_names):
    """Chuyển ner_tags từ list[int] (mặc định của CoNLL) sang list[str]"""
    # Ép buộc chuyển đổi, giải quyết vấn đề Key Error
    example["ner_tags_str"] = [tag_names[tag_id] for tag_id in example["ner_tags"]]
    return example

# Áp dụng hàm chuyển đổi nhãn và tạo cột nhãn string mới
dataset = dataset.map(lambda x: convert_to_string_tags_safe(x, tag_names))

train_sentences = dataset["train"]["tokens"]
# DÙNG CỘT NHÃN MỚI
train_tags = dataset["train"]["ner_tags_str"] 
val_sentences = dataset["validation"]["tokens"]
val_tags = dataset["validation"]["ner_tags_str"] 

# 3. Xây dựng Từ điển (Vocabulary)
word_to_ix = {PAD_TOKEN: 0, UNK_TOKEN: 1}
vocab_index = 2

for sentence in train_sentences:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = vocab_index
            vocab_index += 1

vocab_size = len(word_to_ix)
output_size = len(tag_to_ix) 

print(f"\nKích thước từ điển (Word Vocabulary): {vocab_size}")
print(f"Kích thước từ điển nhãn (Tag Vocabulary): {output_size}")
print(f"Index của {PAD_TOKEN} (dùng cho padding câu): {word_to_ix[PAD_TOKEN]}")
print(f"ID nhãn padding (dùng cho ignore_index): {PAD_LABEL_ID}")

# --- Task 2: Tạo PyTorch Dataset và DataLoader ---

print("\n--- Task 2: Tạo PyTorch Dataset và DataLoader ---")

class NERDataset(Dataset):
    """Lớp Dataset cho bài toán NER"""
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.tags = tags # Dữ liệu tags giờ đã là list[str]
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.unk_idx = self.word_to_ix[UNK_TOKEN]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Câu (Word -> Index)
        sentence = self.sentences[idx]
        sentence_indices = [self.word_to_ix.get(word, self.unk_idx) for word in sentence]
        
        # Nhãn (Tag string -> Index số)
        tags = self.tags[idx]
        # Bằng cách sử dụng cột nhãn mới (ner_tags_str) và đảm bảo map,
        # lỗi KeyError: 0 sẽ được giải quyết vì `tag` luôn là một chuỗi (ví dụ: 'O').
        tag_indices = [self.tag_to_ix[tag] for tag in tags] 

        return torch.tensor(sentence_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)

# 2. Viết hàm collate_fn
def collate_fn(batch):
    sentences = [item[0] for item in batch]
    tags = [item[1] for item in batch]

    # Đệm câu: sử dụng index của <PAD>
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=word_to_ix[PAD_TOKEN])
    
    # Đệm nhãn: sử dụng giá trị đặc biệt PAD_LABEL_ID
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=PAD_LABEL_ID)
    
    return sentences_padded, tags_padded

# 1. Tạo Dataset
train_dataset = NERDataset(train_sentences, train_tags, word_to_ix, tag_to_ix)
val_dataset = NERDataset(val_sentences, val_tags, word_to_ix, tag_to_ix)

# 2. Tạo DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Số lượng batch trong Training DataLoader: {len(train_dataloader)}")
print(f"Số lượng batch trong Validation DataLoader: {len(val_dataloader)}")

# --- Task 3: Xây dựng Mô hình RNN ---

print("\n--- Task 3: Xây dựng Mô hình RNN ---")

class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, rnn_type='LSTM'):
        super(SimpleRNNForTokenClassification, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.rnn_type = rnn_type

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("rnn_type phải là 'RNN', 'GRU', hoặc 'LSTM'")

        self.hidden2tag = nn.Linear(hidden_dim, output_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        rnn_out, _ = self.rnn(embeds) 
        tag_space = self.hidden2tag(rnn_out)
        return tag_space

# Khởi tạo mô hình
model = SimpleRNNForTokenClassification(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_size=output_size,
    rnn_type='LSTM' 
)

print(f"Mô hình được sử dụng: {model.rnn_type}")
print(model)

model.to(device)

# --- Task 4: Huấn luyện Mô hình ---

print("\n--- Task 4: Huấn luyện Mô hình ---")

# 1. Khởi tạo
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID) 

# Bắt đầu vòng lặp huấn luyện
for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    total_loss = 0
    model.train() 

    for sentences, tags in train_dataloader:
        sentences, tags = sentences.to(device), tags.to(device)
        
        # 1. Xóa gradient cũ
        optimizer.zero_grad()

        # 2. Forward pass
        tag_scores = model(sentences)
        
        # 3. Tính loss
        loss = loss_function(
            tag_scores.view(-1, output_size), 
            tags.view(-1)
        )
        total_loss += loss.item()

        # 4. Backward pass
        loss.backward()

        # 5. Cập nhật trọng số
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    end_time = time.time()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Time: {end_time - start_time:.2f}s")


# --- Task 5: Đánh giá Mô hình ---

print("\n--- Task 5: Đánh giá Mô hình ---")

# 1. Viết hàm evaluate
def evaluate(model, dataloader, pad_label_id, ix_to_tag, device):
    model.eval() 
    total_correct = 0
    total_tokens = 0
    
    all_preds = []
    all_true_labels = []

    with torch.no_grad(): 
        for sentences, tags in dataloader:
            sentences, tags = sentences.to(device), tags.to(device)
            
            tag_scores = model(sentences) 
            predictions = torch.argmax(tag_scores, dim=2) 
            
            mask = tags != pad_label_id
            
            correct_preds = (predictions == tags) & mask
            total_correct += correct_preds.sum().item()
            total_tokens += mask.sum().item()
            
            # Chuẩn bị cho seqeval
            for i in range(sentences.shape[0]):
                true_labels_idx = tags[i][mask[i]].cpu().numpy()
                true_labels_str = [ix_to_tag[idx] for idx in true_labels_idx]
                all_true_labels.append(true_labels_str)

                preds_idx = predictions[i][mask[i]].cpu().numpy()
                preds_str = [ix_to_tag[idx] for idx in preds_idx]
                all_preds.append(preds_str)

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    report = classification_report(all_true_labels, all_preds, zero_division=0)
    
    return accuracy, report

# Thực hiện đánh giá
val_accuracy, val_report = evaluate(model, val_dataloader, PAD_LABEL_ID, ix_to_tag, device)

print(f"\nĐộ chính xác (Token-level Accuracy) trên tập Validation: {val_accuracy * 100:.2f}%")
print("\nBáo cáo chi tiết (Seqeval - Entity-level metrics):\n")
print(val_report)

# Hàm dự đoán cho một câu mới
def predict_sentence(sentence_str, model, word_to_ix, ix_to_tag, unk_idx, device):
    """
    Nhận vào chuỗi câu, tiền xử lý và in ra dự đoán NER (bao gồm cả nhãn 'O').
    """
    model.eval()
    
    tokens = sentence_str.split()
    indices = [word_to_ix.get(token, unk_idx) for token in tokens]
    
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    
    with torch.no_grad():
        tag_scores = model(input_tensor) 
        predictions = torch.argmax(tag_scores, dim=2).squeeze(0).cpu().numpy()
        predicted_tags = [ix_to_tag[idx] for idx in predictions]

    print("\n--- Kết quả dự đoán NER trên câu ví dụ (Tất cả nhãn) ---")
    results = []
    
    # THAY ĐỔI Ở ĐÂY: Duyệt và in tất cả các cặp (từ, nhãn)
    for word, tag in zip(tokens, predicted_tags):
        results.append((word, tag))
        print(f"| {word:10} | {tag:8} |") # Định dạng bảng cho dễ nhìn

    return results

# Ví dụ dự đoán (Task 5, phần cuối)
example_sentence = "VNU University is located in Hanoi"
predict_sentence(
    example_sentence, 
    model, 
    word_to_ix, 
    ix_to_tag, 
    word_to_ix[UNK_TOKEN], 
    device
)