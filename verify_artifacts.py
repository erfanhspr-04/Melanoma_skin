import json
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model
MODEL_DIR = Path("saved_models")
MODELS = ["EfficientNetB3_final.keras", "MobileNetV2_final.keras"]
HISTORIES = ["history_efficient_transfer.pkl", "history_efficient_ft.pkl",
"history_mobile.pkl", "history_mobile_ft.pkl"]
def main() -> None:
print("=== ("=== بررسی مدلها
for name in MODELS:
path = MODEL_DIR / name
if not path.exists():
print(f"[!] پیدا نشد : {path}")
continue
model = load_model(path)
print(f"[OK] {name:30s} | پارامتر : {model.count_params():>12,} "
f"| حجم : {path.stat().st_size / 1024**2:6.2f} MB")
print("\n=== ("=== بررسی تاریخچههای آموزش
for name in HISTORIES:
path = MODEL_DIR / name
if not path.exists():
print(f"[!] پیدا نشد : {path}")
continue
with path.open("rb") as f:
keys = sorted(pickle.load(f).keys())
print(f"[OK] {name:30s} | کلیدها : {keys}")
print("\n=== ("=== بررسی فایل پیکربندی
cfg = MODEL_DIR / "project_config.json"
if cfg.exists():
with cfg.open(encoding="utf-8") as f:
config = json.load(f)
print(f"[OK] کلاسها : {config.get('class_names')}")
print(f"[OK] لایهٔ Grad-CAM: {config.get('efficientnet_gradcam_layer')}")
else:
print("[!] project_config.json (".پیدا نشد
if __name__ == "__main__":
main()