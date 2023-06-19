# Traffic_sign_predict
Nhận biết biển báo giao thông bằng tập dữ liệu gtsrb

** Bước 1** Cài đặt thư viện

`> python install -r requiments.txt`

** Bước 2** Tải tệp data về

[data](https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip)

** Bước 3** Chạy file traffic để nó huấn luyện data

`> python traffic.py gtsrb model.h5`

** Bước 4** Chạy file traffic_predict để nó dự đoán biển báo giao thông

`>python traffic_predict.py model.h5 data`

