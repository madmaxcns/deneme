"""

def make_prediction_with_model(test_answers):
    # test_answers içinden gerekli verileri çıkarın ve bir dizi haline getirin
    answers_list = [test_answers.soru_1, 
                    test_answers.soru_2, 
                    test_answers.soru_3, 
                    test_answers.soru_4, 
                    test_answers.soru_5, 
                    test_answers.soru_6, 
                    test_answers.soru_7, 
                    test_answers.soru_8, 
                    test_answers.soru_9, 
                    test_answers.soru_10, 
                    test_answers.soru_11, 
                    test_answers.soru_12, 
                    test_answers.soru_13, 
                    test_answers.soru_14, 
                    test_answers.soru_15, 
                    test_answers.soru_16, 
                    test_answers.soru_17, 
                    test_answers.soru_18, 
                    test_answers.soru_19,
                    test_answers.soru_20]
    # Modelin tahmin yapması için gerekli formata dönüştürün
    answers_array = np.array(answers_list).reshape(1, -1)
    # Tahmini yapın
    prediction = loaded_model_test_sorular.predict(answers_array)
    return prediction[0]




"""
