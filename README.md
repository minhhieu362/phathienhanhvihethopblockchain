# He thong phat hien hanh vi sinh vien bang AI va Blockchain

Prototype realtime dung webcam de phat hien:

- Ngu gat: mat nham lau hoac dau cui qua nguong.
- Dung dien thoai: YOLO detect `person` va `cell phone`, sau do kiem tra dien thoai gan nguoi.
- Mat tap trung: quay dau/nhin lech qua nguong trong nhieu giay.
- Roi khoi vi tri: khong thay mat hoac nguoi trong vai giay.
- Gio tay phat bieu: MediaPipe Pose kiem tra co co tay cao hon vai trong mot khoang thoi gian ngan.
- Dung day: MediaPipe Pose kiem tra chieu cao body/posture thay doi so voi baseline dang ngoi.
- Noi chuyen: FaceMesh kiem tra chuyen dong mieng, ket hop quay dau/nguoi ben canh de suy doan.
- Noi dung am thanh: trinh duyet nghe micro, chuyen giong noi thanh text, backend phan loai hoc tap/noi chuyen rieng va bat den khi la noi chuyen rieng.

Moi lan co vi pham, he thong se chup anh, luu SQLite, tao SHA256 tu `behavior + timestamp + image bytes`, va ghi hash xac minh vao Ethereum mock ledger hoac Ethereum Sepolia. Khi co ha tang blockchain that, co the cau hinh adapter Ethereum bang bien moi truong.

## Cai dat

```powershell
pip install -r requirements.txt
```

Neu dashboard hien `Pose OFF`, cai lai dung ban MediaPipe co `mp.solutions.pose`:

```powershell
pip uninstall -y mediapipe
pip install mediapipe==0.10.14
python -c "import mediapipe as mp; print(hasattr(mp, 'solutions'), hasattr(mp.solutions, 'pose'))"
```

Ket qua can la `True True`, sau do chay lai `python app.py`.

## Chay dashboard

```powershell
python app.py
```

Mo trinh duyet:

```text
http://localhost:5000
```

Dashboard gom webcam realtime, trang thai hien tai, danh sach vi pham, anh minh chung, SHA256 hash va Ethereum tx.

## Du lieu luu tru

- SQLite: `logs/behavior.db`
- Anh minh chung: `logs/snapshots/`
- Ethereum mock ledger: `logs/ethereum_ledger.jsonl`

Bang SQLite `behavior_logs` luu:

```text
id, behavior, timestamp, image_path, hash, ethereum_tx, status
```

API kiem tra hash:

```text
GET /api/events/<id>/verify
```

## Cau hinh nhanh

Co the thay doi bang bien moi truong:

```powershell
$env:CAMERA_INDEX="0"
$env:DROWSY_EYE_SECONDS="0.75"
$env:DROWSY_HEAD_SECONDS="1.1"
$env:DROWSY_SIGNAL_GRACE_SECONDS="0.30"
$env:EYE_CLOSED_EAR="0.21"
$env:EYE_CLOSED_ADAPTIVE_RATIO="0.68"
$env:HEAD_DOWN_RATIO="0.58"
$env:HEAD_DOWN_DELTA="0.07"
$env:DISTRACTED_SECONDS="2"
$env:DISTRACTED_YAW_RATIO="0.14"
$env:DISTRACTED_FACE_LOST="1"
$env:LEFT_SEAT_SECONDS="3"
$env:RAISED_HAND_SECONDS="0.5"
$env:RAISED_HAND_MARGIN="0.04"
$env:STANDING_SECONDS="1"
$env:STANDING_BODY_SPAN_RATIO="0.48"
$env:STANDING_BASELINE_DELTA="0.12"
$env:STANDING_PERSON_HEIGHT_RATIO="0.62"
$env:STANDING_PERSON_ASPECT_RATIO="1.55"
$env:USE_BOX_STANDING="0"
$env:POSE_VISIBILITY="0.55"
$env:TALKING_SECONDS="1"
$env:TALKING_MOUTH_RATIO="0.10"
$env:TALKING_MOUTH_DELTA="0.035"
$env:TALKING_TURN_YAW_RATIO="0.10"
$env:AUDIO_TALKING_HOLD_SECONDS="7"
$env:AUDIO_PRIVATE_HOLD_SECONDS="10"
$env:AUDIO_PRIVATE_THRESHOLD="1"
$env:PHONE_CONFIDENCE="0.35"
$env:YOLO_BEHAVIOR_CONFIDENCE="0.45"
$env:YOLO_MODEL="models/classroom_behavior_yolo11n_best.pt"
$env:PERSON_YOLO_MODEL="yolo11n.pt"
python app.py
```

Mac dinh app uu tien model custom da train tu dataset `Classroom Behavior.v1i.yolov11`
neu ton tai tai `models/classroom_behavior_yolo11n_best.pt`. Model nay nhan dien
`Eating`, `Sleeping`, va `Usingmobile`; trong app `Usingmobile` duoc map thanh
`Dung dien thoai`. App van dung them `PERSON_YOLO_MODEL` mac dinh `yolo11n.pt`
de dem `person` va ho tro phat hien `cell phone`.

Muon quay lai model YOLO11 COCO goc de phat hien `person` va `cell phone`:

```powershell
$env:YOLO_MODEL="yolo11n.pt"
$env:USE_YOLO_BEHAVIOR_CLASSES="0"
python app.py
```

Tat YOLO neu may yeu:

```powershell
$env:USE_YOLO="0"
python app.py
```

## Am thanh va den LED

Dashboard co nut `Start mic`. Khi bam nut nay, trinh duyet se xin quyen micro va dung Web Speech API de tao transcript tieng Viet. Backend nhan text qua `POST /api/audio_transcript`, phan loai:

- `Hoc tap`: noi dung co tu khoa lien quan bai hoc, bai tap, mon hoc, kiem tra.
- `Noi chuyen rieng`: noi dung co tu khoa nhu game, di choi, mang xa hoi, hen, mua ban.
- `Chua ro`: chua du du kien de ket luan.

Neu phan loai la `Noi chuyen rieng`, den canh bao se bat trong `AUDIO_PRIVATE_HOLD_SECONDS`. Neu noi dung la hoc tap hoac chua ro, den khong bat.

Mac dinh app hien den ao tren dashboard. Neu chay tren Raspberry Pi va muon dieu khien LED that qua GPIO:

```powershell
pip install gpiozero
$env:LED_GPIO_PIN="17"
$env:LED_ACTIVE_HIGH="1"
python app.py
```

Noi chan duong LED vao GPIO da cau hinh qua dien tro han dong, chan am ve GND.

Neu dung Arduino ket noi USB, upload sketch Arduino nhan lenh `ON`/`OFF` qua Serial, sau do cau hinh cong COM:

```powershell
pip install pyserial
$env:LED_SERIAL_PORT="COM3"
$env:LED_SERIAL_BAUD="9600"
python app.py
```

Thay `COM3` bang cong Arduino dang hien trong Arduino IDE.

## Ethereum Sepolia

Mac dinh app dung mock ledger de demo. De gui transaction that len Sepolia, deploy smart contract co function:

```solidity
function storeBehaviorHash(string memory behaviorId, string memory hashData, uint256 timestamp) public;
```

Sau do cau hinh:

```powershell
$env:ETHEREUM_ENABLED="1"
$env:ETHEREUM_RPC_URL="https://sepolia.infura.io/v3/<project-id>"
$env:ETHEREUM_PRIVATE_KEY="<private-key-test-wallet>"
$env:ETHEREUM_CONTRACT_ADDRESS="<contract-address>"
$env:ETHEREUM_CONTRACT_ABI='<abi-json>'
$env:ETHEREUM_CHAIN_ID="11155111"
python app.py
```

Luu y: che do server-side nay dung private key test wallet tren backend. Neu muon ky bang vi nguoi dung, dung luong MetaMask ben duoi.

## MetaMask

Dashboard co nut `Connect MetaMask` va nut `Gui hash` tren tung dong su kien. Luong nay de trinh duyet ky giao dich bang MetaMask, sau do app luu tx hash vao cot Ethereum.

Deploy smart contract co function:

```solidity
function storeBehaviorHash(string memory behaviorId, string memory hashData, uint256 timestamp) public;
```

Cau hinh contract cho frontend:

```powershell
$env:METAMASK_CONTRACT_ADDRESS="<contract-address>"
$env:METAMASK_CHAIN_ID="11155111"
python app.py
```

Neu ABI contract khac ABI mac dinh, them:

```powershell
$env:METAMASK_CONTRACT_ABI='<abi-json>'
```

Tren trinh duyet, cai MetaMask, chuyen sang dung chain da cau hinh, bam `Connect MetaMask`, roi bam `Gui hash` o su kien muon ghi len blockchain.
