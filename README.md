# NEXA BIOTECH — Research Agent

Gemini AI destekli, otomatik biyoteknoloji araştırma ajanı.

## Render'a Deploy Etme (Ücretsiz)

### 1. GitHub'a Yükle
```bash
git init
git add .
git commit -m "nexa agent init"
git remote add origin https://github.com/KULLANICI/nexa-agent.git
git push -u origin main
```

### 2. Render'da Yeni Servis Oluştur
- https://render.com → **New → Web Service**
- GitHub reponuzu bağlayın

### 3. Render Ayarları
| Alan | Değer |
|------|-------|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120` |
| **Runtime** | Python 3 |

### 4. Environment Variables
Render Dashboard → **Environment** sekmesi:
```
GEMINI_API_KEY = AIzaSy...buraya_gemini_key...
```

### 5. Deploy
Deploy butonuna basın. ~2 dakikada hazır.

---

## Lokal Çalıştırma
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="AIza..."
python app.py
# → http://localhost:5000
```

## Özellikler
- **8 Kategori:** CRISPR, Nootropics, Peptidler, Longevity, Epigenetik, mRNA, Exosomes, Out-of-Box
- **Duplicate engelleme** — aynı bulgu asla iki kez kaydedilmez
- **Yenilik skoru** — 1-10 arası novelty rating
- **Oto Mod** — 30 dakikada bir otomatik tarama
- **LocalStorage** — veriler tarayıcıda kalıcı
- **google-genai SDK** — `from google import genai` ile Gemini 2.0 Flash

## Model Notu
`gemini-2.0-flash` kullanılıyor (Gemini 2.5 çıktığında `app.py`'de model adını güncelleyin).
