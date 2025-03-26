from fastapi import FastAPI, Request, HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .db import SessionLocal, create_tables
from .models import HastaMetin, HastaMetinCreate, HastaSorular, HastaSorularCreate, HastaGozlem, HastaGozlemCreate
import pickle  # pickle modülünü import edin
import numpy as np
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session
from .models import HastaMetin, HastaSorular 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = 300  # FastText 300 vektör boyutu
hidden_size = 64
output_size = 2 # 0 ve 1 olduğundan dolayı

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Static dosyalar ve templates için yapılandırma
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Model ve vektörleştirici dosyalarının yolları
ml_model_path = 'C:/Users/servet.badem/Desktop/project_root/ml_models/ml_model/logistic_regression_model_pickle.pkl'
model_path = 'C:/Users/servet.badem/Desktop/project_root/ml_models/nlp_model/fasttext_torch_model.pth'
vectorizer_path = 'C:/Users/servet.badem/Desktop/project_root/ml_models/nlp_model/cc.tr.300.bin'

# fasttext vectorizer
from fasttext import load_model
fasttext_model = load_model(vectorizer_path)
def get_text_vectors(texts, model):
    vectors = []
    for text in texts:
        vector = np.mean([model.get_word_vector(word) for word in text.split()], axis=0)
        vectors.append(vector)
    return np.array(vectors)

# torch model
loaded_checkpoint = torch.load(model_path)
loaded_model = SimpleNNModel(input_size, hidden_size, output_size)
loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.001)
loaded_criterion = nn.CrossEntropyLoss()
loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
loaded_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
loaded_criterion.load_state_dict(loaded_checkpoint['criterion_state_dict'])
loaded_epoch = loaded_checkpoint['epoch']
loaded_model.train()

with open(ml_model_path, 'rb') as file:
    loaded_model_test_sorular = pickle.load(file)

# Veritabanı bağlantısı için bağımlılık
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Uygulama başlangıç işlemleri
@app.on_event("startup")
async def startup():
    create_tables()

################################################################################
"""
test_metni = "hayaller kurarak evleniyorsun hepsi gidiyor o kişi sana zarar veriyor"
test_vector = get_text_vectors([test_metni], fasttext_model)

X_test = torch.tensor(test_vector, dtype=torch.float32)

with torch.no_grad():
    model_output = loaded_model(X_test)

predicted_class = torch.argmax(model_output, dim=1).item()

print(f"Tahmin Edilen Sınıf: {predicted_class}")
"""
################################################################################


# Paragraf analizi fonksiyonu
def analyze_paragraph(paragraph: str):
    test_vector = get_text_vectors([paragraph], fasttext_model)
    X_test = torch.tensor(test_vector, dtype=torch.float32)

    with torch.no_grad():
        model_output = loaded_model(X_test)
        probabilities = F.softmax(model_output, dim=1)  # Softmax ile olasılıkları hesapla

    predicted_class = torch.argmax(probabilities, dim=1).item()
    max_proba = torch.max(probabilities).item()  # En yüksek olasılığı al

    return predicted_class, max_proba

# En son kullanıcı paragrafını almak için fonksiyon
def get_latest_user_paragraph(db: Session):
    latest_entry = db.query(HastaMetin).order_by(HastaMetin.sira_numarasi.desc()).first()
    return latest_entry.paragraf if latest_entry else None

def get_latest_user_test(db: Session, hasta_no: int):
    # hasta_no parametresine göre en son doldurulan testi al
    latest_test = db.query(HastaSorular).filter(HastaSorular.hasta_no == hasta_no).order_by(HastaSorular.sira_numarasi.desc()).first()
    return latest_test if latest_test else None

# Route tanımlamaları
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/hasta_metin/")
async def create_hasta_metin(hasta_metin: HastaMetinCreate, db: Session = Depends(get_db)):
    db_hasta = HastaMetin(
        hasta_no=hasta_metin.hasta_no, 
        paragraf=hasta_metin.paragraf,
        cinsiyet=hasta_metin.cinsiyet
    )
    db.add(db_hasta)
    db.commit()
    db.refresh(db_hasta)
    return db_hasta

@app.get("/questionnaire")
async def get_questionnaire(request: Request):
    return templates.TemplateResponse("questionnaire.html", {"request": request})

@app.post("/hasta_sorular/")
async def create_hasta_sorular(sorular: HastaSorularCreate, db: Session = Depends(get_db)):
    db_sorular = HastaSorular(**sorular.dict())
    db.add(db_sorular)
    db.commit()
    db.refresh(db_sorular)
    return db_sorular

@app.get("/get-results")
async def get_results():
    try:
        db = SessionLocal()
        user_paragraph = get_latest_user_paragraph(db)
        predicted_class, proba = analyze_paragraph(user_paragraph)
        
        return {"prediction": str(predicted_class), "proba": str(proba), "user_paragraph": user_paragraph}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict-text/")
async def predict_text_endpoint(text: str):
    try:
        prediction = analyze_paragraph(text)
        return {prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

###########################################################################################
# TEST SONUCLARI TAHMİN MODELİ
###########################################################################################
def make_prediction_with_model(user_input):
    user_input_array = np.array(user_input).reshape(1, -1)
    predicted_proba = loaded_model_test_sorular.predict_proba(user_input_array)[0]
    predicted_class = np.argmax(predicted_proba)
    max_proba = predicted_proba[predicted_class]
    return predicted_class, max_proba

@app.post("/predict-test/")
async def predict_test_result(hasta_no: int, db: Session = Depends(get_db)):
    latest_test = get_latest_user_test(db, hasta_no)
    if latest_test:
        user_input = [latest_test.soru_1,
                      latest_test.soru_2,
                      latest_test.soru_3,
                      latest_test.soru_4,
                      latest_test.soru_5,
                      latest_test.soru_6,
                      latest_test.soru_7,
                      latest_test.soru_8,
                      latest_test.soru_9,
                      latest_test.soru_10,
                      latest_test.soru_11,
                      latest_test.soru_12,
                      latest_test.soru_13,
                      latest_test.soru_14,
                      latest_test.soru_15,
                      latest_test.soru_16,
                      latest_test.soru_17,
                      latest_test.soru_18,
                      latest_test.soru_19,
                      latest_test.soru_20]
        predicted_class, proba = make_prediction_with_model(user_input)
        return predicted_class
    else:
        raise HTTPException(status_code=404, detail="Test bulunamadı veya veri alınırken bir sorun oluştu")

@app.get("/api/get-test-results")
async def get_test_results(db: Session = Depends(get_db)):
    latest_test = db.query(HastaSorular).order_by(HastaSorular.sira_numarasi.desc()).first()
    
    if latest_test:
        user_input = [
            latest_test.soru_1, latest_test.soru_2, latest_test.soru_3,
            latest_test.soru_4, latest_test.soru_5, latest_test.soru_6,
            latest_test.soru_7, latest_test.soru_8, latest_test.soru_9,
            latest_test.soru_10, latest_test.soru_11, latest_test.soru_12,
            latest_test.soru_13, latest_test.soru_14, latest_test.soru_15,
            latest_test.soru_16, latest_test.soru_17, latest_test.soru_18,
            latest_test.soru_19, latest_test.soru_20
        ]
        predicted_class, max_proba = make_prediction_with_model(user_input)
        predicted_class = str(predicted_class)
        max_proba = str(max_proba)

        return {"testPrediction": predicted_class, "proba": max_proba}
    else:
        raise HTTPException(status_code=404, detail="Test bulunamadı veya veri alınırken bir sorun oluştu")

@app.post("/save_predictions/")
async def save_predictions(prediction_data: HastaGozlemCreate, db: Session = Depends(get_db)):
    new_gozlem = HastaGozlem(
        hasta_no=prediction_data.hasta_no,
        paragraf_sonuc=prediction_data.paragraf_sonuc,
        test_sonuc=prediction_data.test_sonuc
    )
    db.add(new_gozlem)
    db.commit()
    db.refresh(new_gozlem)
    return new_gozlem


@app.get("/results")
async def get_results_page(request: Request):
    return templates.TemplateResponse("results.html", {"request": request})

@app.get("/dashboard")
async def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/statistics")
async def get_statistics():
    try:
        db = SessionLocal()

        # Toplam kişi sayısını hesapla
        toplam_kisi = db.query(HastaMetin).count()

        # Erkek ve kadın sayısını hesapla
        erkek_sayisi = db.query(HastaMetin).filter(HastaMetin.cinsiyet == 'Erkek').count()
        kadin_sayisi = db.query(HastaMetin).filter(HastaMetin.cinsiyet == 'Kadın').count()

        # Hasta erkek ve kadın sayısını hesapla
        hasta_erkek = db.query(HastaGozlem).join(HastaMetin, HastaGozlem.hasta_no == HastaMetin.hasta_no).filter(HastaMetin.cinsiyet == 'Erkek', HastaGozlem.test_sonuc == "1").count()
        hasta_kadin = db.query(HastaGozlem).join(HastaMetin, HastaGozlem.hasta_no == HastaMetin.hasta_no).filter(HastaMetin.cinsiyet == 'Kadın', HastaGozlem.test_sonuc == "1").count()

        statistics = {
            "hasta_toplam": hasta_erkek + hasta_kadin,
            "toplam_kisi": toplam_kisi,
            "erkek_sayisi": erkek_sayisi,
            "kadın_sayisi": kadin_sayisi,
            "hasta_erkek": hasta_erkek,
            "hasta_kadin": hasta_kadin
        }
        return statistics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))