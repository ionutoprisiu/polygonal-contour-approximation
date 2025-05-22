# Aproximarea poligonală a contururilor obiectelor
Oprisiu Ionut Daniel
Grupa 30233

## Implementation

### 1. Structura Proiectului
Proiectul este organizat în mai multe fișiere, fiecare având un rol specific:

Fișiere Principale:
- main.cpp - Fișierul principal care rulează aplicația
- image_processor.h - Definiția funcțiilor de procesare
- image_processor.cpp - Implementarea funcțiilor de procesare

### 2. Ce Face Fiecare Fișier?

main.cpp
- Încărcare imagine de intrare
- Procesare imagine
- Afișare rezultate în ferestre separate
- Vizualizare diferite niveluri de aproximare

image_processor.h
- Declarații funcții preprocesare
- Declarații funcții detectare contururi
- Declarații funcții aproximare poligonală
- Declarații funcții afișare rezultate

image_processor.cpp
- preprocessImage() - Preprocesare imagine
- detectContours() - Detectare contururi
- approximateContours() - Aproximare cu epsilon specific
- approximateContoursWithLevel() - Aproximare cu nivel predefinit
- showResults() - Afișare rezultate

### 3. Cum Funcționează Codul?

Pasul 1: Preprocesare
1. Conversie la grayscale
2. Aplicare filtru Gaussian
3. Returnare imagine procesată

Pasul 2: Detectare Contururi
1. Aplicare algoritm Canny
2. Extragere contururi
3. Returnare lista contururi

Pasul 3: Aproximare Poligonală
Varianta 1 (approximateContours):
1. Calcul epsilon
2. Aplicare algoritm de aproximare
3. Returnare imagine cu contururi simplificate

Varianta 2 (approximateContoursWithLevel):
1. Calcul epsilon bazat pe nivel
2. Aplicare algoritm de aproximare
3. Returnare imagine cu contururi simplificate

Pasul 4: Afișare
1. Creare ferestre separate
2. Afișare imagini
3. Așteptare input utilizator

### 4. Parametrii Importanți

Preprocesare
- kernelSize = 5

Detectare Contururi
- threshold1 = 50
- threshold2 = 150

Aproximare
- Nivel 1: epsilon = 0.02 * arcLength
- Nivel 2: epsilon = 0.04 * arcLength
- Nivel 3: epsilon = 0.06 * arcLength
- Epsilon specific: 0.02

### 5. Structura Datelor

Tipuri de Date
- Mat: Imagine OpenCV
- vector<Point>: Contur
- vector<vector<Point>>: Lista contururi
- vector<Vec4i>: Ierarhie contururi

### 6. Validări Implementate

Preprocesare
- Verificare imagine goală
- Verificare kernel size impar

Detectare Contururi
- Verificare imagine validă
- Verificare threshold-uri valide

Aproximare
- Verificare epsilon pozitiv
- Verificare nivel valid

### 7. Afișare Rezultate

Configurare Ferestre
- WINDOW_NORMAL
- Dimensiuni: 400x300
- Titluri descriptive
- Gestionare excepții

### 8. Optimizări

Performanță
- Minimizare copieri imagini
- Reutilizare structuri date
- Operații vectorizate
- Eliberare automată memorie 