SV cần làm các việc và báo cáo bằng slide theo trình tự sau:

Nhóm SV dùng các files tín hiệu đã được thu âm sẵn trong folder “TinHieuHuanLuyen” để thử nghiệm tìm ngưỡng tối ưu, và folder “TinHieuKiemThu” để báo cáo so sánh 2 thuật toán.
Tìm hiểu lý thuyết và cài đặt 02 thuật toán phân đoạn tín hiệu thu âm thành tiếng nói và khoảng lặng (dùng tìm kiếm nhị phân và histogram). 
Chú ý: Độ dài tối thiểu của 1 khoảng lặng là 300 ms (dùng điều kiện này để loại bỏ các khoảng lặng “ảo” có chiều dài quá ngắn).

Các TLTK: 
[1] CS425 Audio and Speech Processing_Hodgkinson_2012:
2.1 Energy-based Speech/Silence discrimination (thuật toán dùng tìm kiếm nhị phân)
[2] A method for silence removal and segmentation of speech signals_Giannakopoulos_2014 (thuật toán dùng histogram).

4. Yêu cầu:
-Mỗi SV trong nhóm cài đặt và demo 01 thuật toán nêu ở trên và báo cáo riêng khi nhóm trình bày. Nhóm cần tổng hợp kết quả cuối cùng để đánh giá so sánh 2 thuật toán.
-SV xuất hình vẽ kết quả hàm STE/MA ngắn hạn (hoặc logSTE/logMA) xếp chồng lên mỗi tín hiệu.
-Chú ý khảo sát ảnh hưởng của mức nhiễu nền (SNR) của môi trường thu âm.
-Các tín hiệu huấn luyện (training data) đóng vai trò giúp xác định bộ tham số tối ưu của thuật toán:
-Hiệu suất của thuật toán đề xuất sẽ được kiểm chứng thực sự trên tập tín hiệu kiểm thử (test data) (GV sẽ upload đầu buổi học tuần sau)
-Để tiết kiệm thời gian chấm thi, mỗi SV chạy script cài đặt riêng task của mình, duyệt qua 4 file tín hiệu kiểm thử và xuất ra 4 figure thể hiện input & output (mỗi figure cho 1 file tín hiệu) trong 01 lần chạy CT duy nhất để GV kiểm tra kết quả.
-Kết quả phân đoạn chuẩn của mỗi file tín hiệu *.wav được chứa trong file *.lab tương ứng. Định dạng của file .lab được mô tả trong file README. SV dùng dữ liệu chuẩn này để đưa ra đánh giá trực quan về độ chính xác của thuật toán như sau: vẽ các biên do thuật toán xuất ra (đường dọc màu xanh) và các biên chuẩn groundtruth (đường dọc màu đỏ) trên đồ thị tín hiệu. SV đánh giá định lượng về độ chính xác bằng cách viết code tính sai số RMSE (Root mean squared error) hoặc MAE (Mean absolute error) giữa các biên chuẩn và biên tìm được (đơn vị đó sai số là miliseconds).
