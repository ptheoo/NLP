# Report Part 1: Tìm hiểu chung về thư viện pytorch và RNNs.
Source code nằm ở lab5_rnn/part1/lab5_rnn_pytorch_intro.py
Sau đây là kết quả sau khi chạy file:


## 1.--- PHẦN 1: KHÁM PHÁ TENSOR ---

[Task 1.1] Tạo Tensor:
Tensor từ list:
 tensor([[1, 2],
        [3, 4]])
Tensor từ NumPy array:
 tensor([[1, 2],
        [3, 4]], dtype=torch.int32)
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])
Random Tensor:
 tensor([[0.2313, 0.0403],
        [0.4298, 0.8258]])

Thông tin của Random Tensor (x_rand):
Shape của tensor: torch.Size([2, 2])
Datatype của tensor: torch.float32
Device lưu trữ tensor: cpu

[Task 1.2] Các phép toán trên Tensor:
Tensor gốc (x_data):
 tensor([[1, 2],
        [3, 4]])
Cộng x_data với chính nó:
 tensor([[2, 4],
        [6, 8]])
Nhân x_data với 5:
 tensor([[ 5, 10],
        [15, 20]])
Nhân ma trận x_data với x_data.T:
 tensor([[ 5, 11],
        [11, 25]])

[Task 1.3] Indexing và Slicing:
Tensor gốc (x_data):
 tensor([[1, 2],
        [3, 4]])
Hàng đầu tiên: tensor([1, 2])
Cột thứ hai: tensor([2, 4])
Giá trị ở hàng thứ hai, cột thứ hai: 4

[Task 1.4] Thay đổi hình dạng Tensor:
Tensor gốc (4, 4):
 tensor([[0.7698, 0.2490, 0.6737, 0.6656],
        [0.9680, 0.7417, 0.7971, 0.9242],
        [0.6694, 0.7127, 0.9941, 0.4925],
        [0.1016, 0.2298, 0.9948, 0.2398]])
Tensor sau khi reshape (16, 1):
 tensor([[0.7698],
        [0.2490],
        [0.6737],
        [0.6656],
        [0.9680],
        [0.7417],
        [0.7971],
        [0.9242],
        [0.6694],
        [0.7127],
        [0.9941],
        [0.4925],
        [0.1016],
        [0.2298],
        [0.9948],
        [0.2398]])
Shape sau khi reshape: torch.Size([16, 1])

## 2.--- PHẦN 2: TỰ ĐỘNG TÍNH ĐẠO HÀM VỚI AUTOGRAD ---
x: tensor([1.], requires_grad=True)
y: tensor([3.], grad_fn=<AddBackward0>)
grad_fn của y: <AddBackward0 object at 0x0000022EBCEDB2E0>
z: tensor([27.], grad_fn=<MulBackward0>)
Đạo hàm của z theo x (dz/dx tại x=1): tensor([18.])

--- PHẦN 3: XÂY DỰNG MÔ HÌNH ĐẦU TIÊN VỚI torch.nn ---

[Task 3.1] Lớp nn.Linear:
Input shape: torch.Size([3, 5])
Output shape: torch.Size([3, 2])
Output (y = xA^T + b):
 tensor([[ 0.2977, -0.1352],
        [ 0.4961, -0.5167],
        [ 0.0377, -0.3053]], grad_fn=<AddmmBackward0>)

[Task 3.2] Lớp nn.Embedding:
Input shape: torch.Size([4])
Output shape: torch.Size([4, 3])
Embeddings (vector 3 chiều cho mỗi chỉ số):
 tensor([[-0.5759,  1.1501,  1.2613],
        [-0.6891,  0.8548, -1.3326],
        [ 1.7238,  0.1797,  0.1541],
        [-0.0629,  0.2488,  1.3241]], grad_fn=<EmbeddingBackward0>)

[Task 3.3] Kết hợp thành một nn.Module (MyFirstModel):
Input data shape: torch.Size([1, 4])
Model output shape (Batch x Seq_len x Output_dim): torch.Size([1, 4, 2])
Model output (ví dụ cho 4 từ):
 tensor([[[-0.5059,  0.2780],
         [-0.4806,  0.2075],
         [-0.3237,  0.2827],
         [-0.3817,  0.5367]]], grad_fn=<ViewBackward0>)

Các tham số của mô hình:
Tên: embedding.weight | Shape: torch.Size([100, 16])
Tên: linear.weight | Shape: torch.Size([8, 16])
Tên: linear.bias | Shape: torch.Size([8])
Tên: output_layer.weight | Shape: torch.Size([2, 8])
Tên: output_layer.bias | Shape: torch.Size([2])

### **Khi thực hiện tính đạo hàm z theo x (Task 2.1), nếu gọi ```z.backward()``` một lần nữa ngay sau lần gọi đầu tiên trong đoạn code trên, nó sẽ gây ra lỗi:**
```RuntimeError: Trying to backward through the graph a second time...```
*Nguyên nhân là do mặc định, PyTorch sẽ giải phóng bộ nhớ đã sử dụng để lưu trữ biểu đồ tính toán (computation graph) ngay sau khi hàm ```.backward()``` được gọi*

*Biểu đồ này là cần thiết để theo dõi các phép toán và tính toán chuỗi đạo hàm ngược. Sau khi tính xong đạo hàm, PyTorch cho rằng bạn đã hoàn thành và dọn dẹp nó để tiết kiệm bộ nhớ (đặc biệt quan trọng trong các mô hình lớn). Khi bạn gọi ```z.backward()``` lần thứ hai, PyTorch không tìm thấy biểu đồ tính toán nữa, do đó không thể thực hiện phép tính đạo hàm ngược và báo lỗi.*

*Tuy nhiên, neus bạn muốn gọi hàm ```.backward()``` nhiều lần thì có thể truyền tham số ```retain_graph=True``` ngay từ lần gọi đầu tiên: ```z.backward(retain_graph=True)```*



Như vậy, ở part 1, chúng ta đã:
- Biết cách tạo và thao tác với Tensor (một trong những thư viện học sâu mạnh mẽ và phổ biến nhất), hiểu được cơ chế tự động tính đạo hàm, và quan trọng nhất là đã tự tay xây dựng một mô-rạng mạng nơ-ron đơn giản bằng ```nn.Module```. 
- Các mô hình cũ sẽ gặp vấn đề với thứ tự của từ chính vì thế, chúng ta cần một mô hình có khả năng xử lý các chuỗi và ghi nhớ thông tin qua từng bước. Đó chính là RNNs.
- Các kiến trúc RNNs phổ biến: LSTM, GRU.
- Bài toán Phân loại Token (Token Classification): Phân loại Token có nghĩa là chúng ta gán một nhãn (label) cho mỗi token (từ) trong một câu đầu vào. Vì RNN có thể tạo ra một đầu ra tại mỗi bước thời gian, nó cực kỳ phù hợp cho loại bài toán này. Một số ví dụ như: POS, SRL....
- Để kết hợp RNN vào bài toán này phân loại Token bao gồm các bước sau:
    1. Embedding: Mỗi từ trong câu đầu vào được chuyển đổi thành một vector
    embedding.
    2. RNN Processing: Chuỗi các vector embedding này được đưa vào RNN. Tại mỗi
    từ, RNN tính toán một trạng thái ẩn h_t, chứa thông tin về từ hiện tại và ngữ cảnh
    từ các từ đứng trước.
    3. Prediction: Trạng thái ẩn h_t (là một vector) được đưa qua một lớp Linear
    (Fully-Connected) để ánh xạ nó sang một vector có số chiều bằng số lượng
    nhãn (ví dụ: 17 nhãn cho bài toán POS).
    4. Softmax: Cuối cùng, hàm Softmax được áp dụng lên vector này để tạo ra một
    phân phối xác suất, cho biết xác suất mỗi nhãn là đúng cho từ hiện tại. Chúng ta
    sẽ chọn nhãn có xác suất cao nhất.