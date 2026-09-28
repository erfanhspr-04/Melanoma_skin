# 🩺 تشخیص ملانوما با یادگیری عمیق و Grad-CAM

**سیستم هوشمند تشخیص ملانوما از روی تصاویر ضایعات پوستی، همراه با مقایسهٔ دو معماری
شبکهٔ عصبی کانولوشنی و توضیح پذیری تصمیم مدل با Grad-CAM و یک اپلیکیشن وب .
---
خلاصه نتایج

مدل	دقت (Accuracy)	Loss	پارامترها	حجم مدل
EfficientNetB3	۹۳.۸۸%	۰.۱۷۰۲	۱۰.۹M	~۸۳ MB
MobileNetV2	۹۲.۱۷%	۰.۲۱۲۵	۲.۴۲M	~۱۱ MB
دیتاست: ۱۰,۶۸۳ تصویر آموزش و ۳,۵۶۴ تصویر اعتبارسنجی (دو کلاس: Melanoma / NotMelanoma)
مدل نهایی: EfficientNetB3 - حساسیت کلاس ملانوما ۹۷.۱% و دقت اعلام ملانوما ۹۰.۵%
توضیح‌پذیری (Explainability): پیاده‌سازی Grad-CAM بر پایه لایه block6f_project_conv
مدل سبک: MobileNetV2 برای اجرا روی موبایل و سیستم‌های کم‌منبع


## 🗂 ساختار پروژه
```
Melanoma_skin/
├── notebooks/
│ ├── 01_mobilenet_training.ipynb # آموزش MobileNetV (مدل سبک) 2
│ └── 02_efficientnet_and_gradcam.ipynb # EfficientNetB3 + Grad-CAM + ارزیابی
├── app.py # اپلیکیشن Streamlit
├── saved_models/
│ ├── README.md # راهنمای دانلود فایلهای مدل
│ └── project_config.json
├── results/
│ ├── model_comparison.csv
│ ├── GradCAM_final_results.csv
│ └── gradcam_samples/ # نمونههای تصویری Grad-CAM
├── docs/ # گزارش PDF و تصاویر مستندات
├── verify_artifacts.py
├── requirements.txt
├── LICENSE
└── README.md
```
## 🚀 نصب و اجرا

```bash
# دریافت کد ( ۱
git clone https://github.com/erfanhspr-04/Melanoma_skin.git
cd Melanoma_skin
# ساخت محیط مجازی (اختیاری ولی توصیهشده) ( ۲
python -m venv venv
venv\Scripts\activate # ویندوز
# source venv/bin/activate # لینوکس / مک
# نصب وابستگیها ( ۳
pip install -r requirements.txt
```
** دانلود فایلهای مدل از اینجا: https://github.com/erfanhspr-04/Melanoma_skin/releases
و قرار دادن آنها در پوشهٔ `saved_models/`.
** **:اجرای اپلیکیشن ( ۵
```bash
python -m streamlit run app.py
```

،سپس در مرورگر، تصویر یک ضایعهٔ پوستی را بارگذاری کن؛ نتیجهٔ طبقهبندی، درصد احتمال
سطح ریسک و نقشهٔ حرارتی Grad-CAM .نمایش داده میشود

## 🧠 روش کار

یادگیری انتقالی** — بدنهٔ شبکه با وزنهای** . 1 ImageNet بارگذاری و فریز میشود؛ فقط سرِ جدید آموزش
.میبیند
۱) تنظیم دقیق** — لایههای انتهایی بدنه با نرخ یادگیری کوچک** . 2 e- .بازآموزی میشوند ( ۵
.قرارداد ورودی** — تصاویر خام (بازهٔ ۰ تا ۲۵۵ ) وارد مدل میشوند؛ پیشپردازش وظیفهٔ خود مدل است** . 3
توضیحپذیری** — نقشهٔ** . 4 Grad-CAM از گرادیان امتیاز کلاس هدف نسبت به نقشهٔ ویژگی لایهٔ کانولوشنی
.محاسبه میشود

## 📊 خروجیها

- `results/model_comparison.csv` — جدول کامل متریکهای دو مدل
- `results/GradCAM_final_results.csv` — نتایج نمونههای بررسیشده با Grad-CAM
- `results/gradcam_samples/` — برای هر نمونه: تصویر اصلی، تصویر بهبودیافته و همپوشانی Grad-CAM
- `docs/` — گزارش کامل پروژه (PDF)

## ⚠️ محدودیتها

دیتاست متعادل است؛ در دنیای واقعی شیوع ملانوما بسیار کمتر است و - Precision در استقرار واقعی افت
.میکند
.مجموعهٔ آزمون مستقل وجود ندارد و ارزیابی روی همان مجموعهٔ اعتبارسنجی انجام شده است -
آموزش روی - CPU .انجام شده و تعداد اپوکها بر همین اساس محدود نگه داشته شده است
- مورد ملانوما (از ۱٬۷۸۳ مورد) بهدرستی شناسایی نشده؛ در کاربرد پزشکی این نوع خطا گرانترین ۱۶۹
.است

## ⚕️ سلب مسئولیت

این پروژه یک کار آموزشی/پژوهشی است و ابزار تشخیص پزشکی محسوب نمیشود. هیچ نتیجهای از این
.سیستم نباید جایگزین نظر پزشک متخصص شود

## 👤 نویسنده

**Erfan Hosseinpoor** — [@erfanhspr-04](https://github.com/erfanhspr-04)

## 📄 مجوز

این پروژه تحت مجوز [MIT](LICENSE) .منتشر شده است
