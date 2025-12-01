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

### Ưu điểm:

- Tốc độ nhanh, Ít tốn tài nguyên: Rất nhanh vì chỉ cần chọn/nối ghép hoặc áp dụng luật đơn giản.

- Đa dạng ngôn ngữ: Dễ dàng điều chỉnh các luật cho các ngôn ngữ khác nhau.

- Độ rõ ràng cao: Âm thanh thường rõ ràng.
*Nhược điểm:*

- Ít tự nhiên: Giọng nói thường có sự "giật cục" hoặc độ biến thiên không tự nhiên ở các điểm nối.

- Khó thêm biểu cảm: Giọng nói nghe đơn điệu, máy móc.

 *Trường hợp sử dụng phù hợp:*

- Các hệ thống yêu cầu hiệu suất cao và tài nguyên thấp (ví dụ: thông báo khẩn cấp, thiết bị nhúng với bộ nhớ hạn chế).

- Các ứng dụng không cần tính tự nhiên tuyệt đối (ví dụ: đọc màn hình cho người khiếm thị với tốc độ cao).