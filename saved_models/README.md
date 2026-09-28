# 📦 پوشهٔ مدلها

فایلهای آموزشدیدهٔ مدل به دلیل حجم، در مخزن گیت قرار نمیگیرند و از بخش
**Releases** :قابل دانلود هستند
👉 https://github.com/erfanhspr-04/Melanoma_skin/releases
فایلهای مورد نیاز ##
| فایل | حجم تقریبی | توضیح |
|---|---|---|
| `EfficientNetB3_final.keras` | ~۸۳ | ٪مگابایت | مدل اصلی پروژه — دقت ۹۳.۸۸
| `MobileNetV2_final.keras` | ~۱۱ | ٪مگابایت | مدل سبک — دقت ۹۲.۱۷
| `history_*.pkl` ( ۱ مگابایت | تاریخچهٔ آموزش دو مدل > | (فایل ۴ |
نحوهٔ استفاده ##
فایلها را از . ۱ Releases .دانلود کن
همه را داخل همین پوشه . ۲ (`saved_models/`) .قرار بده
:ساختار نهایی باید اینطور باشد . ۳
```
saved_models/
├── EfficientNetB3_final.keras
├── MobileNetV2_final.keras
├── project_config.json
├── history_efficient_transfer.pkl
├── history_efficient_ft.pkl
├── history_mobile.pkl
└── history_mobile_ft.pkl
```
بررسی سلامت فایلها ##

:پس از دانلود، در ریشهٔ پروژه اجرا کن
```bash
python verify_artifacts.py
```