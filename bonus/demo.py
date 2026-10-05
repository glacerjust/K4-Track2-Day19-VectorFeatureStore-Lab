"""Demo kịch bản 5 câu hỏi minh họa hoạt động của HybridMemoryAgent.

Chạy: python bonus/demo.py
Yêu cầu: Exit 0, in ra context được lắp ghép cho 5 câu hỏi mẫu.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Thêm root vào sys.path để import app
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent


def main() -> int:
    print("=" * 70)
    print("🚀 BONUS CHALLENGE DEMO — HYBRID MEMORY AI AGENT")
    print("=" * 70)

    # 1. Khởi tạo Agent
    print("[1] Khởi tạo HybridMemoryAgent (Qdrant in-memory + Feast Profile)...")
    agent = HybridMemoryAgent(in_memory=True)
    user_id = "u_001"

    # 2. Nạp một số mẩu ký ức (Episodic Memory) mẫu cho người dùng u_001
    print("[2] Nạp các mẩu ký ức ban đầu vào Episodic Memory...")
    seed_memories = [
        "Đã đọc bài viết hướng dẫn cấu hình Kubernetes Horizontal Pod Autoscaler (HPA) theo CPU metrics.",
        "Đã lưu ghi chú về kiến trúc Zero-Trust và cấu hình xác thực JWT OAuth2 cho hệ thống Cloud API Gateway.",
        "Đã hoàn thành khóa học tối ưu chi phí hạ tầng AWS EKS và tự động mở rộng node groups khi traffic tăng.",
        "Đã đánh dấu tài liệu về triển khai mô hình học sâu Transformer và kỹ thuật RAG trên môi trường Kubernetes.",
        "Đã đọc bài nghiên cứu so sánh PostgreSQL vs DynamoDB cho bài toán lưu trữ event log dung lượng lớn.",
    ]
    for mem in seed_memories:
        agent.remember(mem, user_id=user_id)
    print(f"  -> Đã nạp thành công {len(seed_memories)} ký ức vào Qdrant.\n")

    # 3. Chạy 5 câu truy vấn kiểm thử theo đúng yêu cầu đề bài
    test_queries = [
        (
            1,
            "Tôi đã đọc gì về Kubernetes?",
            "Query đơn giản: Trích xuất trúng các bài viết về Kubernetes từ Vector Store.",
        ),
        (
            2,
            "Recommend đọc gì tiếp cho tôi?",
            "Query cần Profile Context: Khai thác topic_affinity từ Feature Store để gợi ý phù hợp sở thích.",
        ),
        (
            3,
            "Tôi đang quan tâm gì gần đây và hoạt động thế nào?",
            "Query cần Fresh Activity: Khai thác queries_last_hour và distinct_topics_24h.",
        ),
        (
            4,
            "Tài liệu về tự động mở rộng hạ tầng?",
            "Query Paraphrase: Không có chữ 'autoscaling/HPA', kiểm chứng khả năng hiểu ngữ nghĩa của Vector Store.",
        ),
        (
            5,
            "Cho tôi summary cloud security kết hợp thói quen đọc của tôi",
            "Query Mixed (Hybrid): Kết hợp cả ký ức tài liệu bảo mật và tốc độ đọc (reading_speed_wpm).",
        ),
    ]

    print("=" * 70)
    print("🎯 BẮT ĐẦU CHẠY 5 TRUY VẤN DEMO")
    print("=" * 70)

    for q_id, query, description in test_queries:
        print(f"\n--- [QUERY #{q_id}] ----------------------------------------------------")
        print(f"❓ Câu hỏi: {query!r}")
        print(f"💡 Mục tiêu: {description}")
        print("-" * 70)
        context = agent.recall(query, user_id=user_id, top_k=2)
        print(context)
        print("-" * 70)

    print("\n" + "=" * 70)
    print("✅ DEMO HOÀN TẤT THÀNH CÔNG (Exit code: 0)!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
