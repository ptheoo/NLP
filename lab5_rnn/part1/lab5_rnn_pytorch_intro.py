import torch
import numpy as np
from torch import nn

# --- PHẦN 1: KHÁM PHÁ TENSOR ---
print("--- PHẦN 1: KHÁM PHÁ TENSOR ---")

# Task 1.1: Tạo Tensor
print("\n[Task 1.1] Tạo Tensor:")
data = [[1, 2], [3, 4]]

# Tạo tensor từ list
x_data = torch.tensor(data)
print(f"Tensor từ list:\n {x_data}")

# Tạo tensor từ NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor từ NumPy array:\n {x_np}")

# Tạo tensor với các giá trị ngẫu nhiên hoặc hằng số
x_ones = torch.ones_like(x_data) # tạo tensor gồm các số 1 có cùng shape với x_data
print(f"Ones Tensor:\n {x_ones}")

# tạo tensor ngẫu nhiên (chuyển dtype để phù hợp với yêu cầu)
x_rand = torch.rand_like(x_data, dtype=torch.float) 
print(f"Random Tensor:\n {x_rand}")

# In ra shape, dtype, và device của tensor
print(f"\nThông tin của Random Tensor (x_rand):")
print(f"Shape của tensor: {x_rand.shape}")
print(f"Datatype của tensor: {x_rand.dtype}")
print(f"Device lưu trữ tensor: {x_rand.device}")


# Task 1.2: Các phép toán trên Tensor
print("\n[Task 1.2] Các phép toán trên Tensor:")
print(f"Tensor gốc (x_data):\n {x_data}")

# 1. Cộng x_data với chính nó
x_sum = x_data.add(x_data)
# Hoặc x_sum = x_data + x_data
print(f"Cộng x_data với chính nó:\n {x_sum}")

# 2. Nhân x_data với 5
x_mul_scalar = x_data * 5
print(f"Nhân x_data với 5:\n {x_mul_scalar}")

# 3. Nhân ma trận x_data với x_data.T (ma trận chuyển vị của nó)
# x_data có shape (2, 2)
x_T = x_data.T 
# Ma trận nhân: (2, 2) @ (2, 2)
x_matmul = x_data @ x_T 
print(f"Nhân ma trận x_data với x_data.T:\n {x_matmul}")


# Task 1.3: Indexing và Slicing
print("\n[Task 1.3] Indexing và Slicing:")
print(f"Tensor gốc (x_data):\n {x_data}")

# 1. Lấy ra hàng đầu tiên
first_row = x_data[0]
print(f"Hàng đầu tiên: {first_row}")

# 2. Lấy ra cột thứ hai (chú ý index 1)
second_column = x_data[:, 1]
print(f"Cột thứ hai: {second_column}")

# 3. Lấy ra giá trị ở hàng thứ hai, cột thứ hai (index [1, 1])
value_1_1 = x_data[1, 1].item() # .item() để lấy giá trị số Python
print(f"Giá trị ở hàng thứ hai, cột thứ hai: {value_1_1}")


# Task 1.4: Thay đổi hình dạng Tensor
print("\n[Task 1.4] Thay đổi hình dạng Tensor:")
original_tensor = torch.rand(4, 4)
print(f"Tensor gốc (4, 4):\n {original_tensor}")

# Biến nó thành một tensor có shape (16, 1)
reshaped_tensor = original_tensor.reshape(16, 1)
# Hoặc: reshaped_tensor = original_tensor.view(16, 1)
print(f"Tensor sau khi reshape (16, 1):\n {reshaped_tensor}")
print(f"Shape sau khi reshape: {reshaped_tensor.shape}")


# --- PHẦN 2: TỰ ĐỘNG TÍNH ĐẠO HÀM VỚI AUTOGRAD ---
print("\n--- PHẦN 2: TỰ ĐỘNG TÍNH ĐẠO HÀM VỚI AUTOGRAD ---")

# Task 2.1: Thực hành với autograd
# Tạo một tensor và yêu cầu tính đạo hàm cho nó
x = torch.ones(1, requires_grad=True)
print(f"x: {x}")

# Thực hiện một phép toán: y = x + 2
y = x + 2
print(f"y: {y}")

# y được tạo ra từ một phép toán có x, nên nó cũng có grad_fn
print(f"grad_fn của y: {y.grad_fn}")

# Thực hiện thêm các phép toán: z = y * y * 3 = 3 * (x+2)^2
z = y * y * 3
print(f"z: {z}")

# Tính đạo hàm của z theo x
# tương đương z.backward(torch.tensor(1.))
z.backward() 

# Đạo hàm được lưu trong thuộc tính .grad
# Ta có z = 3 * (x+2)^2 => dz/dx = 6 * (x+2). Với x=1, dz/dx = 6 * (1+2) = 18
print(f"Đạo hàm của z theo x (dz/dx tại x=1): {x.grad}")

# --- PHẦN 3: XÂY DỰNG MÔ HÌNH ĐẦU TIÊN VỚI torch.nn ---
print("\n--- PHẦN 3: XÂY DỰNG MÔ HÌNH ĐẦU TIÊN VỚI torch.nn ---")

# Task 3.1: Lớp nn.Linear
print("\n[Task 3.1] Lớp nn.Linear:")
# Khởi tạo một lớp Linear biến đổi từ 5 chiều -> 2 chiều
linear_layer = torch.nn.Linear(in_features=5, out_features=2)

# Tạo một tensor đầu vào mẫu (3 mẫu, mỗi mẫu 5 chiều)
input_tensor = torch.randn(3, 5) 

# Truyền đầu vào qua lớp linear
output = linear_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"Output (y = xA^T + b):\n {output}")


# Task 3.2: Lớp nn.Embedding
print("\n[Task 3.2] Lớp nn.Embedding:")
# Khởi tạo lớp Embedding cho một từ điển 10 từ, mỗi từ biểu diễn bằng vector 3 chiều
embedding_layer = torch.nn.Embedding(num_embeddings=10, embedding_dim=3)

# Tạo một tensor đầu vào chứa các chỉ số của từ (ví dụ: một câu)
# Các chỉ số phải nhỏ hơn 10 (từ 0 đến 9)
input_indices = torch.LongTensor([1, 5, 0, 8])

# Lấy ra các vector embedding tương ứng
embeddings = embedding_layer(input_indices)

print(f"Input shape: {input_indices.shape}")
print(f"Output shape: {embeddings.shape}")
print(f"Embeddings (vector 3 chiều cho mỗi chỉ số):\n {embeddings}")


# Task 3.3: Kết hợp thành một nn.Module
print("\n[Task 3.3] Kết hợp thành một nn.Module (MyFirstModel):")

class MyFirstModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MyFirstModel, self).__init__()
        # Định nghĩa các lớp (layer) bạn sẽ dùng
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Linear layer 1: embedding_dim -> hidden_dim
        self.linear = nn.Linear(embedding_dim, hidden_dim) 
        self.activation = nn.ReLU() # Hàm kích hoạt ReLU
        # Linear layer 2 (Output): hidden_dim -> output_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, indices):
        # Định nghĩa luồng dữ liệu đi qua các lớp
        # 1. Lấy embedding (shape: batch_size, seq_len, embedding_dim)
        embeds = self.embedding(indices) 
        
        # 2. Truyền qua lớp linear và hàm kích hoạt
        # Linear layer hoạt động trên chiều cuối cùng (embedding_dim)
        hidden = self.activation(self.linear(embeds)) 

        # 3. Truyền qua lớp output
        # Output layer hoạt động trên chiều cuối cùng (hidden_dim)
        output = self.output_layer(hidden) 
        return output

# Khởi tạo và kiểm tra mô hình
VOCAB_SIZE = 100
EMBEDDING_DIM = 16
HIDDEN_DIM = 8
OUTPUT_DIM = 2 # Ví dụ: 2 class để phân loại

model = MyFirstModel(
    vocab_size=VOCAB_SIZE, 
    embedding_dim=EMBEDDING_DIM, 
    hidden_dim=HIDDEN_DIM, 
    output_dim=OUTPUT_DIM
)

# Tạo dữ liệu đầu vào mẫu: 1 câu (batch_size=1), gồm 4 từ (seq_len=4)
input_data = torch.LongTensor([[1, 2, 5, 9]]) 
# Shape của input_data: (1, 4)

output_data = model(input_data)
# Shape của output_data sẽ là (batch_size, seq_len, output_dim) = (1, 4, 2)

print(f"Input data shape: {input_data.shape}")
print(f"Model output shape (Batch x Seq_len x Output_dim): {output_data.shape}")
print(f"Model output (ví dụ cho 4 từ):\n {output_data}")

# In ra các tham số (parameters) của mô hình
print("\nCác tham số của mô hình:")
for name, param in model.named_parameters():
    print(f"Tên: {name} | Shape: {param.shape}")