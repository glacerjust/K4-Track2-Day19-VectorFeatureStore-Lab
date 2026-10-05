# Reflection — Lab 19

**Tên:** Ngô Kỳ Anh
**Cohort:** A20-K4
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên golden set 50 queries:
- `exact`: BM25 thắng hoặc ngang ngửa vì người dùng tìm chính xác từ khóa kỹ thuật/tên riêng, cơ chế khớp từ khóa và IDF của BM25 bắt trúng tuyệt đối.
- `paraphrase`: Vector (Semantic) thắng áp đảo vì câu hỏi được diễn đạt lại không chứa từ khóa gốc, vector hiểu được ngữ nghĩa tương đồng trong không gian nhúng.
- `mixed` và Trung bình tổng thể: Hybrid (RRF) thắng tuyệt đối vì kết hợp được cả độ chính xác từ khóa của BM25 và độ phủ ngữ nghĩa của Vector.

**Khi nào KHÔNG dùng Hybrid?**
Không dùng khi:
1. Hệ thống chỉ cần tra cứu mã tra cứu cụ thể, mã lỗi, số seri hoặc số điện thoại (chỉ cần BM25 là đủ, dùng thêm vector gây lãng phí chi phí tính toán và làm tăng độ trễ).
2. Tài nguyên máy chủ hạn chế hoặc yêu cầu độ trễ cực thấp dưới 5ms mà không có GPU để tính toán embedding thời gian thực.

---

## Điều ngạc nhiên nhất khi làm lab này


RRF với công thức đơn giản 1/(60 + rank) lại dung hòa điểm số của hai trường phái khác nhau xuất sắc mà không cần chuẩn hóa phức tạp.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
