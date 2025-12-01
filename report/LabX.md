# Tổng quan về bài toán Text to Speech (TTS)
Bài toán Text-to-Speech (TTS) là quá trình tự động tổng hợp giọng nói từ văn bản đầu vào. Mục tiêu là tạo ra giọng nói nhân tạo không chỉ dễ hiểu mà còn tự nhiên, biểu cảm và mang đặc trưng giọng nói (âm sắc, tốc độ, ngữ điệu) mong muốn.

Các thách thức chính mà nghiên cứu TTS hướng tới bao gồm:

- Tính tự nhiên (Naturalness): Giọng nói tổng hợp phải giống giọng người thật, bao gồm ngữ điệu (prosody), trọng âm, và sự ngắt nghỉ.

- Tính biểu cảm (Expressiveness): Khả năng tổng hợp giọng nói với các cảm xúc (vui, buồn, giận,...) và phong cách nói khác nhau (kể chuyện, đọc tin, đối thoại).

- Hiệu suất và tài nguyên: Tốc độ tổng hợp nhanh (real-time synthesis) và yêu cầu tài nguyên tính toán (CPU/GPU) thấp.

- Tính đa ngôn ngữ/đa giọng nói (Multi-lingual/Multi-speaker): Khả năng hoạt động trên nhiều ngôn ngữ hoặc tổng hợp giọng nói của nhiều người khác nhau (giọng nam, nữ, trẻ em,...) chỉ với một mô hình.

- Dữ liệu thấp (Low-Resource/Few-Shot): Khả năng tổng hợp giọng nói chất lượng cao chỉ với một lượng rất nhỏ dữ liệu âm thanh của giọng nói mục tiêu.

---

# Các Phương pháp triển khai và Tình hình nghiên cứu
Các phương pháp triển khai TTS chính có thể được phân loại thành ba cấp độ (Levels) như bạn đã đề cập, phản ánh sự phát triển từ các hệ thống dựa trên luật đến các mô hình học sâu (Deep Learning) phức tạp.

## Level 1: TTS dựa trên Luật/Nối ghép (Rule-Based/Concatenative TTS)
Đây là phương pháp truyền thống, khởi đầu của TTS.

### Cách triển khai:
- Concatenative (Nối ghép): Ghi âm một cơ sở dữ liệu lớn các đơn vị giọng nói (âm vị, âm tiết, từ, hoặc nửa âm vị). Khi cần tổng hợp, hệ thống sẽ chọn và nối ghép các đơn vị này lại với nhau theo các luật ngữ âm.
- Rule-Based (Dựa trên Luật): Sử dụng các luật ngữ âm học và ngôn ngữ học để chuyển văn bản thành chuỗi các tham số âm học (như tần số cơ bản - $F_0$, phổ âm,...) sau đó dùng bộ mã hóa (vocoder) để tạo âm thanh.

*Ưu điểm:*

- Tốc độ nhanh, Ít tốn tài nguyên: Rất nhanh vì chỉ cần chọn/nối ghép hoặc áp dụng luật đơn giản.

- Đa dạng ngôn ngữ: Dễ dàng điều chỉnh các luật cho các ngôn ngữ khác nhau.

- Độ rõ ràng cao: Âm thanh thường rõ ràng.

*Nhược điểm:*

- Ít tự nhiên: Giọng nói thường có sự "giật cục" hoặc độ biến thiên không tự nhiên ở các điểm nối.

- Khó thêm biểu cảm: Giọng nói nghe đơn điệu, máy móc.

 *Trường hợp sử dụng phù hợp:*

- Các hệ thống yêu cầu hiệu suất cao và tài nguyên thấp (ví dụ: thông báo khẩn cấp, thiết bị nhúng với bộ nhớ hạn chế).

- Các ứng dụng không cần tính tự nhiên tuyệt đối (ví dụ: đọc màn hình cho người khiếm thị với tốc độ cao).

## Level 2: TTS dựa trên Học sâu Tham số (Parametric Deep Learning TTS)

Đây là thế hệ TTS đầu tiên áp dụng Học sâu, cải thiện đáng kể tính tự nhiên.

### Cách triển khai: 
Sử dụng các mô hình Mạng nơ-ron sâu (DNNs, RNNs, LSTMs) để học ánh xạ từ đặc trưng văn bản (từ, âm vị) sang các tham số âm học (như Mel-spectrogram, $F_0$, độ dài âm vị). Sau đó, một bộ mã hóa thống kê hoặc DNN (như WaveNet, WaveRNN) sẽ chuyển các tham số này thành dạng sóng âm thanh.
*Ví dụ:* Tacotron (biến đổi từ Seq2Seq), DeepVoice.

*Ưu điểm:*
- Tính tự nhiên cao hơn Level 1: Giọng nói mượt mà, ngữ điệu tự nhiên hơn nhiều.
- Tính tùy chỉnh cao: Cho phép người dùng ghi âm một lượng dữ liệu vừa đủ để tinh chỉnh (finetune) mô hình (Personalized Voice Cloning - PVC) để tạo giọng nói cá nhân với ít tài nguyên hơn Level 3.

*Nhược điểm:*
- Yêu cầu dữ liệu lớn: Cần tập dữ liệu giọng nói chất lượng cao, lớn để huấn luyện mô hình cơ sở.
- Pipeline phức tạp: Thường bao gồm nhiều bước (Text analysis $\rightarrow$ Acoustic Model $\rightarrow$ Vocoder).
- Thách thức đa ngôn ngữ: Khó đảm bảo tính đa dạng ngôn ngữ nếu không có dữ liệu phong phú.

*Trường hợp sử dụng phù hợp:*
- Các ứng dụng trợ lý ảo (Virtual Assistants) như Siri, Google Assistant.
- Lồng tiếng cho video, sách nói yêu cầu chất lượng cao
- Tạo giọng nói cá nhân (PVC) khi người dùng có sẵn dữ liệu và muốn tối ưu hóa tài nguyên so với Few-shot.

## Level 3: TTS Đầu cuối Few-Shot/Zero-Shot (Few-Shot/Zero-Shot End-to-End TTS)

Đây là hướng nghiên cứu hiện tại, tập trung vào khả năng tùy biến giọng nói nhanh chóng với dữ liệu thấp.
### Cách triển khai:
 Mô hình học sâu Đầu cuối (End-to-End) (ví dụ: VITS, EATS, Meta's Voicebox, Google's Lyra) được huấn luyện để trực tiếp ánh xạ từ văn bản sang dạng sóng âm thanh hoặc Mel-spectrogram, thường kết hợp bộ mã hóa (vocoder) vào kiến trúc chính.

- Few-Shot/Zero-Shot Voice Cloning: Mô hình được thiết kế để tách biệt nội dung nói (content) khỏi đặc trưng giọng nói (style/speaker identity). Bằng cách cung cấp chỉ vài giây (Few-Shot) hoặc không giây (Zero-Shot) âm thanh giọng nói tham chiếu (Reference Audio), mô hình có thể tổng hợp văn bản mới bằng giọng nói tham chiếu đó.

*Ưu điểm:*

- Tính tự nhiên và Biểu cảm cao nhất: Thường cho kết quả gần giống giọng người thật nhất.

- Tốn ít công sức cho người dùng: Chỉ cần cung cấp một đoạn âm thanh mẫu (ví dụ: 3-5 giây) để tạo ra giọng nói mong muốn.

- Pipeline đơn giản hơn: Mô hình tích hợp, giảm bớt các thành phần riêng biệt

*Nhược điểm:*

- Tốn tài nguyên tính toán: Mô hình rất phức tạp, đòi hỏi GPU mạnh và thời gian huấn luyện lâu.

- Khó kiểm soát: Do tính chất đầu cuối, đôi khi khó debug hoặc tinh chỉnh từng thành phần riêng biệt (như ngữ điệu).

- Rủi ro Deepfake: Khả năng tạo ra giọng nói giả mạo cao.

*Trường hợp sử dụng phù hợp:*

- Tạo nội dung linh hoạt: Thay đổi giọng nói nhanh chóng cho podcast, game, quảng cáo.

- Ứng dụng cá nhân hóa tối đa: Tạo giọng nói theo yêu cầu của bất kỳ ai ngay lập tức.

- Nghiên cứu tiên tiến: Tập trung vào biểu cảm, đa ngôn ngữ, và zero-shot learning.

# Cách các nghiên cứu tạo Pipeline để Tối thiểu hoá Nhược điểm
Các nghiên cứu hiện đại thường kết hợp các hướng tiếp cận để tối ưu hóa đồng thời hiệu suất, tính tự nhiên và tài nguyên:
| **Thách thức**                                   | **Giải pháp Pipeline Hiện tại**                                                                                                                                         | **Kết quả / Mục tiêu**                                                                                                  |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **Hiệu suất / Tài nguyên** (Nhược điểm Level 2, 3) | **Knowledge Distillation**: Huấn luyện mô hình nhỏ (Student) bắt chước mô hình lớn, chậm (Teacher).                                                                      | Giữ được độ tự nhiên của mô hình lớn nhưng chạy **nhanh hơn**, **nhẹ hơn**, phù hợp thiết bị tài nguyên thấp.            |
| **Tính tự nhiên / Biểu cảm** (Nhược điểm Level 1, 2) | **Vocoder Diffusion / GANs**: Dùng các vocoder hiện đại như **HiFi-GAN**, **WaveFlow**, **Diffusion** thay thế vocoder thống kê truyền thống.                            | Âm thanh mượt, giàu biểu cảm, **gần như không phân biệt được với người thật**.                                            |
| **Dữ liệu thấp / Few-shot** (Nhược điểm Level 1, 2) | **Speaker Embedding**: Dùng bộ mã hóa giọng nói (Speaker Encoder) để trích đặc trưng giọng từ 1 đoạn audio tham chiếu và dùng để điều kiện hóa mô hình TTS.             | Tạo giọng mới chỉ với **vài giây âm thanh**, hỗ trợ đa giọng mà **không cần huấn luyện lại**.                            |
| **Đạo đức nghiên cứu**                            | **Watermarking**: Nhúng tín hiệu không nghe được vào sóng âm để đánh dấu âm thanh do AI tạo.                                                                             | Chống **deepfake**, cho phép truy xuất nguồn gốc giọng nói tổng hợp, đảm bảo an toàn và minh bạch.                      |
