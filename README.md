# Aproximare Poligonală a Contururilor Obiectelor

Acest proiect implementează o soluție pentru aproximarea poligonală a contururilor obiectelor din imagini color (BMP, 24 biți/pixel).

## Structura Proiectului

```
.
├── docs/               # Documentație și documente de proiect
├── include/           # Header files
├── src/              # Fișiere sursă
├── build/            # Director pentru build
└── CMakeLists.txt    # Configurare CMake
```

## Cerințe

- CMake (versiune 3.10 sau mai nouă)
- OpenCV
- Compilator C++ cu suport pentru C++17

## Compilare

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Executabilul va fi generat în `build/bin/`.

## Utilizare

```bash
./bin/polygonal_approximation <imagine_input> <imagine_output>
```

## Funcționalități

1. Preprocesare imagine:
   - Conversie la grayscale
   - Filtrare Gaussiană pentru reducerea zgomotului

2. Extragere contururi:
   - Detectare muchii folosind algoritmul Canny
   - Extragere contururi folosind findContours

3. Aproximare poligonală:
   - Aplicarea algoritmului Ramer-Douglas-Peucker
   - Parametrul epsilon este calculat automat ca 2% din perimetrul conturului

## Fișiere Importante

- `include/image_processor.h` - Header pentru procesarea imaginilor
- `src/image_processor.cpp` - Implementarea procesării imaginilor
- `src/main.cpp` - Programul principal
- `CMakeLists.txt` - Configurare build
- `docs/` - Documentație și documente de proiect

## Autor

Oprisiu Ionut Daniel
Grupa 30233

## Design Choices și Implementare

### 1. Algoritm de Aproximare Poligonală
Am ales algoritmul Ramer-Douglas-Peucker (RDP) pentru următoarele motive:
- Eficiență computațională: O(n log n) complexitate
- Păstrează forma generală a obiectului
- Implementare robustă în OpenCV
- Control precis asupra nivelului de aproximare prin parametrul epsilon

### 2. Preprocesare Imagine
- **Conversie la grayscale**: Simplifică procesarea și reduce complexitatea
- **Filtrare Gaussiană**: 
  - Kernel size = 5: Echilibru între reducerea zgomotului și păstrarea detaliilor
  - Sigma = 0: Valoare implicită care oferă rezultate bune pentru majoritatea imaginilor

### 3. Detectare Contururi
- **Algoritm Canny**:
  - Threshold1 = 50, Threshold2 = 150: Valori standard care funcționează bine pentru majoritatea imaginilor
  - Raport 1:3 între threshold-uri pentru detectarea optimă a muchiilor
- **findContours**:
  - RETR_EXTERNAL: Extrage doar contururile externe
  - CHAIN_APPROX_SIMPLE: Reduce numărul de puncte păstrând forma

### 4. Aproximare Poligonală
- **Parametrul epsilon**:
  - Valoare bazată pe perimetrul conturului (0.02 * arcLength)
  - Niveluri de aproximare (1-3) pentru flexibilitate
  - Nivel 1: Aproximare precisă
  - Nivel 2: Aproximare medie
  - Nivel 3: Aproximare simplificată

### 5. Optimizări Implementate
- **Preprocesare eficientă**:
  - Conversie directă la grayscale
  - Filtrare Gaussiană optimizată
- **Detectare contururi eficientă**:
  - Parametri Canny optimizați
  - Extragere doar a contururilor externe
- **Aproximare eficientă**:
  - Calcul epsilon bazat pe perimetru
  - Niveluri de aproximare predefinite

### 6. Structura Codului
- **Modularizare**:
  - Funcții separate pentru fiecare etapă
  - Parametri configurabili
  - Cod ușor de extins
- **Reutilizare**:
  - Funcții helper pentru operații comune
  - Structuri de date clare

### 7. Performanță
- **Timp de procesare**:
  - Preprocesare: O(n) unde n este numărul de pixeli
  - Detectare contururi: O(n log n)
  - Aproximare: O(n log n)
- **Memorie**:
  - Utilizare eficientă a memoriei
  - Copiere minimă a datelor

### 8. Extensibilitate
- Ușor de adăugat noi funcționalități
- Parametri ușor de ajustat
- Suport pentru diferite formate de imagine

## Parametri Configurabili
- `kernelSize`: Dimensiunea kernel-ului pentru filtrare Gaussiană
- `threshold1`, `threshold2`: Threshold-uri pentru Canny
- `epsilon`: Pragul de precizie pentru aproximare
- `level`: Nivelul de aproximare (1-3) 