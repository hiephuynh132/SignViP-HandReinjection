# Chạy bằng Docker

Dockerfile dùng PyTorch `2.7.1-cuda12.8-cudnn9-runtime`. Host cần NVIDIA driver
và NVIDIA Container Toolkit. Repository có thể đặt ở bất kỳ thư mục nào.

## Build

Chạy từ root repository:

```bash
docker build -t signvip-handreinjection:paper .
```

Build cần Internet để tải base image/packages, không tải model weight và không
chạy training. `.dockerignore` loại data/weight/output khỏi build context.
Layer cuối kiểm tra source và import trainer/evaluator.

## Chuẩn bị mount

Tải weight theo `WEIGHT_PATHS.md`, dùng cấu trúc trong README:

```text
data/
  rwth/{train,dev,test}/
  hamer/{train,dev,test}/
  detail_masks/{train,dev,test}/
  dwpose_keypoints/test/
weights/
  stable-diffusion-v1-5/
  optimized_vae/vae/
  proposed/best/
```

```bash
mkdir -p outputs .cache
```

Container đọc `/data`, `/weights`, ghi `/outputs`, cache tại `/cache`. Đây là
tên mount trong container, đường dẫn host do bạn chọn qua `src=`. Các lệnh dưới
chạy từ root repository. Tránh symlink trỏ ra ngoài cây thư mục được mount;
hãy bind mount thư mục đích thật.

## Kiểm tra GPU 1

```bash
docker run --rm --gpus 'device=1' signvip-handreinjection:paper \
  python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
```

GPU vật lý 1 có logical index 0 trong container. Giữ `GPU_ID=0` bên trong;
đổi `device=1` sang `device=0` nếu muốn dùng GPU vật lý 0.

## Evaluate paper checkpoint

Thử một video trước:

```bash
docker run --rm --gpus 'device=1' --shm-size=8g \
  --mount "type=bind,src=$PWD/data,dst=/data,readonly" \
  --mount "type=bind,src=$PWD/weights,dst=/weights,readonly" \
  --mount "type=bind,src=$PWD/outputs,dst=/outputs" \
  --mount "type=bind,src=$PWD/.cache,dst=/cache" \
  -e OUTPUT_DIR=/outputs/eval_smoke \
  signvip-handreinjection:paper \
  bash scripts/eval.sh --max-samples 1 --save-previews 1
```

Evaluate toàn bộ test: dùng cùng lệnh, bỏ `-e OUTPUT_DIR=...` và hai flag
`--max-samples 1 --save-previews 1`. Kết quả ở `outputs/eval_proposed/` trên host.
Cache model có thể được tải lần đầu; khi offline cần chuẩn bị cache trước.

## Train proposal

Phần này chỉ là tham khảo: cần checkpoint khởi tạo C5.14 không được phát hành
trong gói inference/evaluation. Chỉ tải best C5.15 không đủ để chạy protocol train này.

```bash
docker run --rm --gpus 'device=1' --shm-size=8g \
  --mount "type=bind,src=$PWD/data,dst=/data,readonly" \
  --mount "type=bind,src=$PWD/weights,dst=/weights,readonly" \
  --mount "type=bind,src=$PWD/outputs,dst=/outputs" \
  --mount "type=bind,src=$PWD/.cache,dst=/cache" \
  signvip-handreinjection:paper bash scripts/train_proposal.sh
```

Fine-tune VAE: giữ nguyên mount, thêm `-e SIGNVIP_WORKSPACE_ROOT=/outputs/vae`
trước tên image và đổi command thành `bash scripts/train_vae.sh`.
Evaluate VAE: đổi command thành `bash scripts/eval_vae.sh`.
Các lệnh chạy foreground; Docker không tự chờ GPU trống.

## Resume Stage I

Full resume cần optimizer/scheduler/scaler; `best/` chỉ phục vụ inference.
Với run ở `outputs/stage1/proposal/<run-id>` trên host, thêm các flag sau
trước tên image trong lệnh train:

```bash
-e SIGNVIP_RUN_DIR=/outputs/stage1/proposal/<run-id> \
-e SIGNVIP_RESUME_FROM_CHECKPOINT=/outputs/stage1/proposal/<run-id>
```

Thay `<run-id>` bằng tên run thật. Trainer tìm checkpoint hoàn chỉnh mới nhất.
Không dùng đường dẫn host làm đường dẫn bên trong container.

## Layout host khác mẫu

Đổi `src=` sang thư mục thật, giữ nguyên `dst=`. Hoặc mount riêng từng folder
vào `/data/rwth`, `/data/hamer`, `/data/detail_masks`, `/data/dwpose_keypoints`.
Không cần sửa source hoặc commit đường dẫn host.

Ngoài Docker, launcher tự tìm data/weight từ root checkout, bất kể thư mục shell.
Override qua `DATA_ROOT`, `HAMER_ROOT`, `MASK_ROOT`, `POSE_ROOT`, `SD15_ROOT`,
`PAPER_VAE_ROOT`, `PAPER_INIT_ROOT`, `CHECKPOINT_DIR`.

## Kiểm tra trước khi đưa lên Git

```bash
python tools/check_release.py
```

Tool kiểm tra đường dẫn workstation, cú pháp Python và checksum source export.
Danh sách đường dẫn riêng để upload weight nằm ngoài repository. Bản public
`WEIGHT_PATHS.md` dùng reference tương đối và ô Drive URL. Các số liệu archived
giữ nguyên; chỉ đường dẫn máy cá nhân được thay thế.

Docker build/runtime chưa được xác nhận trên host soạn bản này vì tài khoản
không truy cập được Docker daemon. Source/config/launcher được kiểm tra độc lập.
Cần build và chạy smoke eval trước khi tuyên bố image tái lập kết quả.

Tham chiếu: [PyTorch image](https://hub.docker.com/r/pytorch/pytorch/tags),
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
[Docker GPU access](https://docs.docker.com/engine/containers/resource_constraints/#gpu).
