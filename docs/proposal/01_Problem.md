# 1. PROBLEM

## 1.1. Technical Problem

### Bối cảnh

Các hệ multi-agent hiện nay thường sử dụng nhiều Large Language Model hoặc gọi nhiều lần tới một model lớn. Cách tiếp cận này tạo ra một số vấn đề:

* Chi phí inference cao.
* Latency tăng nhanh theo số agent.
* Nhiều agent cạnh tranh GPU memory và KV cache.
* Các agent trao đổi thông tin lặp lại.
* Context của mỗi agent ngày càng dài.
* Shared memory thường chỉ là vector database đơn giản.
* Việc lưu, chia sẻ và truy xuất memory chưa được tối ưu theo chi phí compute.
* Scheduler của inference engine không hiểu dependency giữa các agent.

Trong khi đó, Small Language Model có thể chạy local trên GPU phổ thông, nhưng năng lực suy luận và context của từng model bị giới hạn. Vì vậy, hệ thống cần nhiều agent chuyên biệt, công cụ ngoài và một cơ chế memory hiệu quả để bù lại giới hạn của từng SLM.

### General Technical Problem

> Làm thế nào xây dựng một hệ Multi-Agent SLM chạy hiệu quả trên một GPU hoặc CPU đơn, trong đó các agent có thể phối hợp, sử dụng private/shared memory và ra quyết định dưới giới hạn latency, VRAM, token budget và compute budget?

Bài toán gồm bốn thành phần liên quan chặt chẽ:

#### Serving

Hệ thống phải phục vụ nhiều logical agent trên cùng một inference runtime.

Các vấn đề cần giải quyết:

* Continuous batching.
* Request scheduling.
* KV-cache allocation.
* Prefix-cache reuse.
* Multi-LoRA serving.
* CPU–GPU workload placement.
* Prefill và decode scheduling.
* Giới hạn VRAM.
* Priority và dependency giữa request.
* Throughput–latency trade-off.

#### Agentic Memory

Mỗi agent cần hai loại memory:

* Private memory: chỉ phục vụ vai trò và quá trình suy luận của agent.
* Shared memory: lưu tri thức chung cho toàn hệ thống.

Shared memory được biểu diễn bằng temporal knowledge graph, trong đó mỗi fact có:

* Entity.
* Relation.
* Timestamp.
* Confidence.
* Provenance.
* Visibility.
* Validity.
* Supporting hoặc contradicting evidence.

Hệ thống phải quyết định:

* Thông tin nào cần lưu.
* Lưu ở private hay shared memory.
* Khi nào cần truy xuất.
* Truy xuất bao nhiêu.
* Khi nào cập nhật hoặc vô hiệu hóa fact.
* Khi nào cần quên hoặc tổng hợp memory.

#### Multi-Agent Coordination

Các agent phải quyết định:

* Agent nào cần được gọi.
* Agent nào chạy trước.
* Agent nào có thể chạy song song.
* Khi nào cần gửi thông tin cho agent khác.
* Khi nào đủ thông tin để hành động.
* Khi nào nên dừng.

#### Learning

Hệ thống cần học các policy thay vì chỉ phụ thuộc vào prompt cố định.

Các policy có thể bao gồm:

* Memory policy.
* Communication policy.
* Tool-use policy.
* Agent-selection policy.
* Scheduling policy.
* Domain decision policy.

### Technical Research Question

> Liệu một hệ Multi-Agent SLM có thể học cách quản lý shared memory, điều phối agent và sử dụng tài nguyên inference hiệu quả hơn các policy thủ công, trong khi vẫn duy trì chất lượng giải quyết nhiệm vụ?

### Technical Objectives

Hệ thống hướng tới:

1. Chạy toàn bộ multi-agent workload trên một GPU consumer 16GB.
2. Dùng một base SLM cho nhiều logical agent.
3. Hỗ trợ private memory và shared temporal knowledge graph.
4. Tối ưu latency, throughput và GPU memory.
5. Giảm communication và context duplication.
6. Huấn luyện policy cho memory, coordination và scheduling.
7. Đánh giá đồng thời task quality và serving efficiency.

### Technical Non-Goals

Giai đoạn đầu không đặt mục tiêu:

* Huấn luyện foundation model từ đầu.
* Thay thế hoàn toàn các hệ distributed inference.
* Xử lý hàng nghìn người dùng đồng thời.
* Xây một knowledge graph tổng quát cho mọi domain.
* Cho phép agent tự do thay đổi ontology.
* Triển khai trực tiếp trong môi trường nghiệp vụ có rủi ro cao.

---

## 1.2. Business Problem

### Domain được chọn

> Multi-Agent Supply Chain Planning and Inventory Coordination

Hệ thống mô phỏng hoạt động của một doanh nghiệp có:

* Nhiều sản phẩm.
* Nhiều kho.
* Nhiều nhà cung cấp.
* Nhiều tuyến vận chuyển.
* Nhu cầu biến động.
* Hạn giao hàng.
* Chi phí lưu kho.
* Rủi ro gián đoạn nguồn cung.

### Business Problem

Các quyết định supply chain thường được thực hiện bởi nhiều bộ phận:

* Demand planning.
* Inventory management.
* Procurement.
* Logistics.
* Risk management.
* Operations coordination.

Mỗi bộ phận có:

* Nguồn dữ liệu riêng.
* Mục tiêu riêng.
* Góc nhìn riêng.
* Giả định riêng.
* Thông tin riêng chưa được chia sẻ.

Điều này dẫn đến:

* Dự báo nhu cầu không đồng nhất.
* Đặt hàng quá nhiều hoặc quá ít.
* Stockout.
* Tồn kho dư thừa.
* Giao hàng trễ.
* Thay đổi supplier chậm.
* Quyết định không tận dụng được lịch sử.
* Lặp lại các lỗi đã xảy ra trước đây.

### Concrete Business Problem

> Khi nhu cầu, tồn kho, supplier và logistics liên tục thay đổi, hệ thống phải phối hợp nhiều agent để đưa ra quyết định bổ sung hàng, chuyển kho, chọn supplier hoặc trì hoãn đơn hàng nhằm giảm tổng chi phí và duy trì service level.

### Các agent nghiệp vụ

#### Demand Agent

* Dự báo nhu cầu.
* Phát hiện demand spike.
* Đưa ra confidence interval.
* So sánh forecast với actual demand.

#### Inventory Agent

* Theo dõi tồn kho.
* Phát hiện stockout risk.
* Tính safety stock.
* Đề xuất chuyển kho.

#### Supplier Agent

* Theo dõi lead time.
* Đánh giá supplier reliability.
* Đề xuất đặt hàng hoặc đổi supplier.

#### Logistics Agent

* Theo dõi shipment.
* Đánh giá tuyến vận chuyển.
* Tính chi phí và thời gian giao hàng.

#### Risk Agent

* Phát hiện disruption.
* Đánh giá mức độ ảnh hưởng.
* Truy xuất các sự kiện tương tự trong quá khứ.

#### Coordinator Agent

* Điều phối các agent.
* Tổng hợp bằng chứng.
* Chọn hành động cuối cùng.
* Kiểm soát token và compute budget.

### Business Actions

Hệ thống có thể đề xuất hoặc thực hiện trong môi trường mô phỏng:

* Đặt thêm hàng.
* Hủy hoặc giảm đơn đặt hàng.
* Chuyển hàng giữa các kho.
* Đổi supplier.
* Chọn tuyến vận chuyển khác.
* Điều chỉnh safety stock.
* Ưu tiên một nhóm đơn hàng.
* Chấp nhận backorder.
* Yêu cầu thêm thông tin.
* Không hành động.

### Business Objective

Tối thiểu hóa:

[
C_{\text{total}} =
C_{\text{stockout}}

* C_{\text{holding}}
* C_{\text{ordering}}
* C_{\text{transport}}
* C_{\text{delay}}
* C_{\text{switching}}
  ]

Trong khi duy trì:

* Service level.
* Order fulfillment rate.
* Supplier reliability.
* Inventory availability.
* Decision explainability.

### Business Success Metrics

* Stockout rate.
* Holding cost.
* Late delivery rate.
* Fill rate.
* Order fulfillment rate.
* Total operating cost.
* Average time-to-decision.
* Số quyết định phải sửa lại.
* Khả năng xử lý disruption lặp lại.
* Mức cải thiện khi sử dụng shared memory.
