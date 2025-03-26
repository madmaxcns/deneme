// Hasta bilgilerini sunucuya göndermek için fonksiyon
function submitHastaForm() {
    var hastaNo = document.getElementById('hastaNo').value;
    var paragraf = document.getElementById('paragraf').value;
    var cinsiyet = document.getElementById('cinsiyet').value; // Cinsiyet bilgisini al

    fetch('/hasta_metin/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            hasta_no: hastaNo,
            paragraf: paragraf,
            cinsiyet: cinsiyet // Cinsiyet bilgisini gönder
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Success:', data);
        localStorage.setItem('hastaNo', hastaNo); // Hasta numarasını localStorage'a kaydet
        window.location.href = '/questionnaire'; // Sorular sayfasına yönlendir
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

// Test sorularını sunucuya göndermek için fonksiyon
function submitTestSorular() {
    var hastaNo = localStorage.getItem('hastaNo') || 'Bilinmeyen Hasta';
    var sorular = {};
    for (var i = 1; i <= 20; i++) {
        var soruCevabi = document.querySelector(`input[name="question${i}"]:checked`);
        sorular[`soru_${i}`] = soruCevabi ? parseInt(soruCevabi.value) : 0;
    }

    fetch('/hasta_sorular/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            hasta_no: hastaNo,
            ...sorular
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Success:', data);
        // NLP ve Test sonuçlarını al ve hasta_gozlem tablosuna kaydet
        fetchPredictionsAndSave(hastaNo);
        window.location.href = '/results'; // Sonuçlar sayfasına yönlendir
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

// NLP ve Test sonuçlarını alıp hasta_gozlem tablosuna kaydetmek için yardımcı fonksiyon
function fetchPredictionsAndSave(hastaNo) {
    Promise.all([
        fetch('/predict-text/').then(res => res.json()),
        fetch('/api/get-test-results').then(res => res.json())
    ])
    .then(([nlpResult, testResult]) => {
        savePredictions(hastaNo, nlpResult.prediction, testResult.testPrediction);
    })
    .catch(error => {
        console.error('Prediction fetch error:', error);
    });
}

// Hasta gözlem verilerini sunucuya kaydetmek için fonksiyon
function savePredictions(hastaNo, paragrafSonuc, testSonuc) {
    fetch('/save_predictions/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            hasta_no: hastaNo,
            paragraf_sonuc: paragrafSonuc,
            test_sonuc: testSonuc
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Kayıt işlemi başarısız');
        }
        console.log('Tahminler başarıyla kaydedildi');
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// Form gönderildiğinde hasta bilgilerini işle
document.getElementById('hastaForm').addEventListener('submit', function(event) {
    event.preventDefault(); 
    submitHastaForm();
});
