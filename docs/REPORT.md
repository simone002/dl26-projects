# Temporal Action Segmentation from Video — Report

- **Group ID**: G36
- **Project ID**: 13

---

## 1. Introduction and Objective

La **segmentazione temporale delle azioni** (*temporal action segmentation*) è il problema di assegnare un'etichetta di azione a ogni singolo frame di un video. Formalmente, dato un video rappresentato come una sequenza di vettori di feature **F** = {**f**₁, **f**₂, …, **f**_T} con **f**_t ∈ ℝ^d, l'obiettivo è produrre una sequenza di predizioni Ŷ = {ŷ₁, ŷ₂, …, ŷ_T} dove ogni ŷ_t ∈ {0, 1, …, C} indica l'azione in corso al frame t, con 0 che rappresenta il background.

A differenza della classificazione di clip (*action recognition*), che produce una sola etichetta per l'intero segmento, la segmentazione temporale richiede una predizione densa frame per frame, catturando sia il contenuto di ogni istante sia la struttura temporale della sequenza. A differenza della rilevazione di azioni (*action detection*), che individua solo i segmenti di interesse, qui si copre l'intera durata del video incluso il background.

| Task | Input | Output | Differenza principale |
|---|---|---|---|
| Action recognition | clip intera | 1 label | nessuna struttura temporale |
| Action detection | video lungo | lista segmenti con timestamp | non copre il background |
| **Action segmentation** | sequenza di frame | 1 label per frame | copertura densa, include background |

### Sfide principali

**Sbilanciamento delle classi.** I clip EGTEA contengono una quota rilevante di background. Senza accorgimenti, un modello tende a predire sempre background ottenendo alta accuracy senza aver appreso le classi di interesse.

**Granularità fine delle classi.** Le 106 classi sono combinazioni di 19 verbi e 51 oggetti. Azioni come *Cut tomato* e *Cut bell pepper* coinvolgono lo stesso movimento su oggetti diversi, rendendone la distinzione difficile da feature RGB.

**Ambiguità dei confini temporali.** La transizione tra due azioni consecutive avviene in modo graduale e le annotazioni riflettono una valutazione soggettiva del momento esatto di cambio.

**Dipendenze temporali a lungo raggio.** Alcune azioni seguono sequenze ricorrenti (aprire → prendere → tagliare → mescolare). Modellare queste dipendenze richiede un contesto che può estendersi su centinaia di frame.

---

## 2. Contribution and Added Value

Il progetto implementa una pipeline completa di segmentazione temporale su EGTEA Gaze+, confrontando quattro architetture con la stessa infrastruttura di training:

- **CNN1D** — baseline convolutiva 1D, punto di riferimento minimo.
- **LSTM bidirezionale** — cattura dipendenze passate e future con stato ricorrente.
- **xLSTM** — variante extended-LSTM con memory mixing;
- **MS-TCN++** — Multi-Stage TCN con convoluzioni dilatate a crescita esponenziale su 4 stage; ogni stage raffina le predizioni dello stage precedente.

Rispetto al semplice uso di codice esistente, i contributi tecnici principali sono:
- Loss combinata CE + Smooth + Boundary con motivazione esplicita per ciascun termine.
- Sliding window in validazione con overlap 50% per una valutazione riproducibile.
- Data augmentation temporale (shift casuale delle label) per ridurre il bias di anticipazione/ritardo.
- **Pipeline di estrazione feature DINOv3**: script di estrazione streaming (`src/utils/extract_dinov3_features.py`) che processa i video raw EGTEA frame per frame con DINOv3 ViT-B (`facebook/dinov3-vitb16-pretrain-lvd1689m`), producendo vettori d = 768 per frame salvati come file `.npy` memory-mapped. L'approccio streaming evita il caricamento dell'intero video in RAM (rischio OOM su video da ~70 GB di frame). Il nuovo datamodule `EGTEADataModuleNpy` legge direttamente i file `.npy`.


---

## 3. Data Used

### Dataset: EGTEA Gaze+

EGTEA Gaze+ è un dataset egocentric acquisito con una telecamera montata sulla testa di 32 soggetti durante sessioni di preparazione pasti in cucina. Il dataset comprende 86 sessioni video per un totale di circa 28 ore a 24 fps.

Le azioni sono annotate con **106 classi** composte da combinazioni verbo-oggetto (es. *Cut tomato*, *Mix egg*, *Pour water*) più una classe background. Sono presenti circa 10.325 istanze di azione distribuite su 7 tipi di pasto.

### Analisi del dataset (split 1, 8.299 clip)

L'analisi statistica su split 1 rivela tre caratteristiche strutturali che motivano le scelte progettuali della pipeline.

**Dominanza del background.** Il background occupa in media il **56%** dei frame per clip; 532 clip (6.4%) sono interamente background e non contengono alcun frame di azione. Questa proporzione giustifica la riduzione del peso del background a `bg_weight = 0.05` nella cross-entropy: senza questo accorgimento, un modello che predice sempre background otterrebbe un'accuracy del 56% senza apprendere nulla sulle 106 classi di interesse.

**Clip prevalentemente brevi.** La durata mediana è **59 frame** (~2.5 s a 24 fps) e il 75° percentile è 103 frame. Con `seq_len = 128`, l'80.8% dei clip viene coperto interamente in una singola finestra; il rimanente 19.2% richiede la sliding window in inferenza. La coda lunga (max 2.801 frame) indica che alcune sessioni sono molto più lunghe della norma.

**Sbilanciamento tra classi foreground.** La distribuzione delle classi segue una coda lunga marcata: la classe più frequente (*Open fridge*, 19.970 frame) supera di **20×** le classi nella metà inferiore della distribuzione. Le classi *Cut tomato*, *Cut cucumber*, *Cut onion* e *Cut carrot* sono tra le più frequenti e visivamente simili tra loro — fatto che spiega la concentrazione di confusioni su queste coppie nell'analisi qualitativa (§5.1).

### Protocollo di valutazione: 3-fold cross-validation

Viene usato il **protocollo ufficiale EGTEA** a 3 fold, che sfrutta tutti e tre gli split ufficiali del dataset:

| Fold | Train | Val / Test |
|---|---|---|
| 1 | split 1 + split 2 | split 3 |
| 2 | split 1 + split 3 | split 2 |
| 3 | split 2 + split 3 | split 1 |

Ogni fold utilizza circa **16.600 clip di training** (due split concatenati via `ConcatDataset`) e circa **8.300 clip di val/test** (un singolo split). Ogni split dispone di un archivio LMDB separato (`TSN-C_3_egtea_action_CE_s{1,2,3}_rgb_model_best_fcfull_hd`). I risultati finali sono la media e la deviazione standard delle metriche sui 3 fold.

### Feature


**DINOv3:** feature estratte dai video raw con **DINOv3 ViT-B** (`facebook/dinov3-vitb16-pretrain-lvd1689m`), producendo vettori **d = 768** per ogni frame. L'estrazione usa lo script `src/utils/extract_dinov3_features.py` in modalità streaming (un batch di frame alla volta) e salva un file `.npy` per video. Questa sorgente non richiede l'archivio LMDB e utilizza un backbone con pre-training su 1.6B immagini (LVD-1689M), potenzialmente più discriminativo per le feature visive fine-grained del dataset.

### Preprocessing e augmentation

**Training — random crop**: ad ogni epoca viene estratta una finestra casuale di `seq_len = 128` frame da ciascun clip, con:
- Gaussian noise sulle feature (`std = 0.05`).
- Feature dropout casuale (`prob = 0.1`).
- Shift temporale casuale delle label di ±5 frame (`prob = 0.5`).

**Validation/Test — sliding window**: l'intera durata del clip viene coperta con finestre di 128 frame e `stride = 64` (overlap 50%), così ogni frame viene valutato.

```
Clip:  |-----------------------------------------------|
       seq_len=128          stride=64

Win 1: |-----128-----|
Win 2:         |-----128-----|
Win 3:                 |-----128-----|
```

---

## 4. Methodology and Architecture

### 4.1 Architetture

Tutti i modelli condividono l'interfaccia `Input: (B, T, feat_dim)` → `Output: (B, T, num_classes)`.

**CNN1D** (`src/models/cnn1d.py`) — stack di convoluzioni 1D con residual connections. Vede un contesto locale limitato dal kernel size, senza memoria a lungo raggio.

**LSTM** (`src/models/lstm.py`) — LSTM bidirezionale con proiezione lineare finale. Controlla il flusso di informazioni con gate discreti (input, forget, output) in modo fisso per posizione. Il contesto bidirezionale permette a ogni frame di vedere sia il passato che il futuro, ma il training è sequenziale e non parallelizzabile.

**xLSTM** (`src/models/xlstm.py`) — extended LSTM con matrici di memoria più espressive (mLSTM con multi-head attention e proiezioni QKV). Usa direttamente la libreria ufficiale NX-AI/xlstm. Estende LSTM sostituendo lo stato vettoriale con una matrice H×H e l'aggiornamento con proiezioni Q, K, V — maggiore capacità ma anche maggiore rischio di overfitting su dataset di medie dimensioni.

**MS-TCN++** (`src/models/mstcn.py`) — implementazione fedele al repository ufficiale MS-TCN2, con architettura asimmetrica tra primo stage e stage successivi:

- **Stage 1 — Prediction_Generation**: doppio flusso di convoluzioni dilatate che operano in parallelo sulla stessa sequenza:
  - *Stream 1*: dilation decrescente (2^(N−1) → 2^0)
  - *Stream 2*: dilation crescente (2^0 → 2^(N−1))
  - I due flussi sono fusi tramite `conv_fusion` (Conv1d(2·hidden, hidden, 1)), poi ReLU + dropout + connessione residuale.
- **Stage 2–4 — Refinement**: convoluzioni dilatate (2^0 → 2^(N−1)) con skip connections; ogni stage prende come input il softmax dell'output del precedente, raffinando progressivamente la segmentazione.

Ha campo recettivo finito determinato dalle dilatazioni, ma completamente parallelizzabile. È l'architettura di riferimento per questo task.

### 4.2 Loss Function

La loss totale combina tre termini:

```
L = L_CE + λ_s · L_smooth + λ_b · L_boundary
```

con `λ_s = 0.15`, `λ_b = 0.3`.

**Cross-Entropy con label smoothing (ε=0.1) e class weights.**
Il peso del background è ridotto a `bg_weight = 0.05` per non dominare il gradiente. Il label smoothing riduce l'overconfidence sulle classi visivamente simili.

**Smooth Loss** — identica all'implementazione ufficiale MS-TCN2:
```
L_smooth = mean( clamp((log_p[t] − log_p[t−1].detach())², max=16) )
```
Il termine `log_p[t−1].detach()` rende la loss **asimmetrica**: il frame precedente è trattato come target fisso, senza propagare il gradiente attraverso di esso. Non viene applicata nessuna maschera GT: la penalità agisce su tutte le transizioni nelle predizioni, scoraggiando il flickering frame-per-frame indipendentemente dall'etichetta GT. Rispetto a una versione simmetrica con maschera, questa formulazione riduce il rischio di sopprimere transizioni corrette nelle zone di confine.

**Boundary Loss.**
Applica una CE con peso maggiore sui frame entro ±3 frame da ogni transizione GT. Incentiva la precisione sul "quando inizia e finisce" ogni azione, con impatto diretto su boundary F1 e F1@50.

### 4.3 Metriche di Valutazione

Quattro metriche complementari coprono aspetti diversi della qualità della segmentazione. La loro implementazione è allineata al file `metrics.py` del repository ufficiale MS-TCN2.

**Frame Accuracy.** Percentuale di frame correttamente classificati sull'intera sequenza, incluso il background. È la metrica più immediata ma può essere gonfiata dalla dominanza del background: un modello che predice sempre background ottiene un'alta accuracy senza aver imparato nulla.

**Edit Score** (basato sulla Levenshtein edit distance). La sequenza di predizioni viene prima collassata in una lista di segmenti contigui (es. `[Cut tomato, Mix egg, Pour water]`), poi si conta il numero minimo di operazioni — inserimento, cancellazione, sostituzione — per trasformare la lista predetta nella lista ground truth:

```
GT:    [Cut tomato → Mix egg → Pour water]           (3 segmenti)
Pred:  [Cut tomato → Put pan → Mix egg → Pour water] (4 segmenti)

Edit distance = 1  →  Edit Score = (1 − 1/4) × 100 = 75%
```

A differenza di F1, non dipende dalla precisione temporale dei confini: cattura se il modello predice le azioni giuste **nel giusto ordine**, ignorando piccoli sfasamenti. È usato come metrica di early stopping perché riflette la struttura sequenziale globale della predizione.

**F1@k** (con k ∈ {10, 25, 50}%). Un segmento predetto è considerato corretto se la sua IoU temporale con il segmento GT corrispondente supera la soglia k%. F1@10 è permissiva (accetta confini approssimativi), F1@50 è severa (richiede almeno il 50% di overlap). La progressione F1@10 → F1@25 → F1@50 mostra quanto la qualità degradi all'aumentare della rigidità sulla localizzazione temporale.

**Boundary F1**. Misura la precisione nella localizzazione dei confini tra azioni: un confine predetto entro una tolleranza di ±2 frame dal confine GT conta come TP. Cattura direttamente la precisione su "quando inizia e finisce" ogni azione, indipendentemente dalla correttezza della classe.

| Metrica | Livello | Cosa penalizza | Sensibile a |
|---|---|---|---|
| Frame Accuracy | frame | ogni frame errato | baseline, background |
| Edit Score | segmento | ordine e numero errati | sequenza globale |
| F1@k | segmento | overlap < k% | precisione temporale |
| Boundary F1 | confine | confini > ±2 frame dal GT | localizzazione esatta |

### 4.4 Training Setup

| Parametro | Valore |
|---|---|
| Ottimizzatore | AdamW |
| Learning rate | 4 × 10⁻⁴ |
| Weight decay | 5 × 10⁻³ |
| Scheduler | CosineAnnealingLR |
| Max epochs | 100 |
| Early stopping | patience 20, monitor `val/edit_score` |
| Gradient clipping | 1.0 |
| Batch size | 64 |
| seq\_len | 128 |

---

## 5. Results and Discussion

I risultati definitivi saranno prodotti con la **3-fold cross-validation** (`train_cv.py`) e le metriche allineate all'implementazione ufficiale MS-TCN2 (frame accuracy su tutti i frame, edit score 0–100, F1@{10,25,50} segment-level, boundary F1).


**Tabella 1**: Risultati su fold 1 (train split 1+2, test split 3) — feature DINOv3 ViT-B.

| Model    | Acc (%) | Edit Score | F1@10 | F1@25 | F1@50 | Boundary F1 |
|----------|:-------:|:----------:|:-----:|:-----:|:-----:|:-----------:|
| CNN1D    | 92.7    | 73.5       | 74.7  | 74.3  | 72.3  | 57.0        |
| LSTM     | 96.6    | 94.8       | 88.9  | 88.7  | 88.2  | 77.9        |
| xLSTM    | 98.0    | 91.4       | 86.7  | 86.6  | 86.3  | 85.3        |
| MS-TCN++ | 96.2    | 94.5       | 88.6  | 88.4  | 87.8  | 77.4        |

*I valori riportati sono relativi al fold 1; l'estensione alla 3-fold cross-validation è in corso.*

**Discussione comparativa.** I quattro modelli si separano in tre fasce nette.

- **CNN1D** è il fanalino di coda su tutte le metriche (edit 73.5, boundary F1 57.0). Il campo recettivo di ~9 frame non basta a modellare le dipendenze temporali: senza memoria, gli errori di classificazione persistono per centinaia di frame (§5.3), facendo crollare sia l'ordine dei segmenti (edit) sia la localizzazione dei confini.

- **LSTM e MS-TCN++** sono sostanzialmente appaiati in testa sulle metriche di segmento (edit 94.8 vs 94.5, F1@10 88.9 vs 88.6, F1@50 88.2 vs 87.8). Entrambi sfruttano un contesto temporale ampio — bidirezionale per l'LSTM, dilatato multi-stage per MS-TCN++ — e mostrano lo stesso comportamento ai confini (segmenti predetti più lunghi del GT, §5.1–5.2). Sono le due architetture di riferimento per questo task.

- **xLSTM** ha un profilo distinto: vince nettamente sulla **frame accuracy** (98.0) e sul **boundary F1** (85.3, contro ~77 degli altri), ma resta indietro su edit (91.4) e F1@k. La lettura è che xLSTM è il più preciso a livello di singolo frame e di confine — predice i bordi quasi esattamente (errore di fine +1.0 frame, §5.4) — ma commette più errori a livello di *struttura sequenziale* dei segmenti, dove edit ed F1@k lo penalizzano. È il modello da preferire se conta la precisione temporale fine, mentre LSTM/MS-TCN++ restano superiori sulla coerenza dell'ordine delle azioni.

In sintesi, nessun modello domina su tutte le metriche: la scelta dipende dall'obiettivo (struttura sequenziale → LSTM/MS-TCN++; precisione di confine → xLSTM), mentre la necessità di memoria temporale a lungo raggio è confermata dal divario netto rispetto alla baseline CNN1D.

### 5.1 Analisi qualitativa — MS-TCN++

L'analisi degli errori sistematici è condotta con `src/evaluation/evaluate.py` sul checkpoint migliore di fold 1 (`fold1-epoch=89-val/edit_score`), valutato su split 3 con sliding window (stride=64) su 2021 clip.

**Errori di confine temporale.** Il modello anticipa l'inizio delle azioni di 9.5 frame in media e ritarda la fine di 10.4 frame, producendo segmenti predetti **più lunghi** di quelli GT. La deviazione standard è alta (57–64 frame) ma la mediana è 0, indicando che la maggioranza delle predizioni è precisa e sono gli outlier sulle azioni rare e lunghe a far salire la media.

| | Media | Std | Mediana |
|---|---|---|---|
| Errore inizio (frame) | −9.5 | 56.9 | 0.0 |
| Errore fine (frame) | +10.4 | 63.7 | 0.0 |

Le classi con errore di confine maggiore sono tutte rare o visivamente ambigue nel tempo:

| Classe | Errore medio (frame) | Segmenti |
|---|---|---|
| Cut olive | 210.1 | 12 |
| Wash strainer | 160.8 | 8 |
| Cut cucumber | 139.2 | 56 |
| Move Around bacon | 111.8 | 41 |
| Cut tomato | 103.2 | 65 |

**Confusioni tra classi.** Le principali confusioni coinvolgono coppie semanticamente vicine:

| GT | Predetto | Frame |
|---|---|---|
| Put cucumber | Cut cucumber | 134 |
| Take tomato | Cut tomato | 62 |
| Operate stove | Move Around bacon | 56 |
| Put sponge | Wash cutting_board | 51 |
| Cut bell_pepper | Cut carrot | 47 |

Il pattern è coerente con la sfida della granularità fine descritta in §1: le coppie *Put/Take cucumber → Cut cucumber* e *Take tomato → Cut tomato* condividono l'oggetto ma differiscono nel verbo, mentre *Cut bell_pepper → Cut carrot* condivide il verbo ma confonde l'oggetto. Anche feature DINOv3 ViT-B (pre-addestrate su 1.6B immagini) non riescono a disambiguare questi casi guardando i singoli frame: il problema non è la qualità delle feature visive ma l'ambiguità intrinseca dell'azione senza contesto temporale sufficiente.

### 5.2 Analisi qualitativa — LSTM

**Errori di confine temporale.** Il BiLSTM mostra un pattern simile a MS-TCN++ ma leggermente più contenuto: anticipa l'inizio di 8.6 frame e ritarda la fine di 9.2 frame, producendo segmenti predetti **più lunghi** dei GT. Questo comportamento è coerente con la natura bidirezionale del modello: avendo accesso al contesto futuro, il modello "vede" l'azione in arrivo e anticipa l'inizio; simmetricamente, non taglia il segmento finché l'attività non è visivamente cessata. La deviazione standard è alta (54–58 frame) ma la mediana è 0, confermando che la maggioranza delle predizioni è accurata e sono gli outlier sulle azioni lunghe a fare salire la media.

| | Media | Std | Mediana |
|---|---|---|---|
| Errore inizio (frame) | −8.6 | 54.4 | 0.0 |
| Errore fine (frame) | +9.2 | 58.3 | 0.0 |

| Classe | Errore medio (frame) | Segmenti |
|---|---|---|
| Cut olive | 201.0 | 12 |
| Wash strainer | 154.8 | 8 |
| Cut cucumber | 117.3 | 54 |
| Move Around bacon | 110.4 | 41 |
| Cut tomato | 101.9 | 65 |

**Confusioni tra classi.** Come MS-TCN++, anche l'LSTM confonde prevalentemente il **verbo** (Take/Put → Cut) mantenendo l'oggetto corretto; le coppie coinvolte sono in larga parte le stesse (Put/Take cucumber → Cut cucumber, Take tomato → Cut tomato), ma su un numero di frame inferiore.

| GT | Predetto | Frame |
|---|---|---|
| Put cucumber | Cut cucumber | 37 |
| Take tomato | Cut tomato | 34 |
| Take cucumber | Cut cucumber | 27 |
| Take cooking_utensil | Mix mixture,eating_utensil | 26 |
| Take bell_pepper | Cut bell_pepper | 26 |

Le coppie Take→Cut condividono lo stesso oggetto ma verbi diversi: visivamente entrambe mostrano una mano che si avvicina all'oggetto, e la distinzione richiede di capire se la mano sta afferrando o tagliando — informazione che può richiedere più contesto temporale di quanto il modello abbia disponibile in una finestra di 128 frame.

### 5.3 Analisi qualitativa — CNN1D

**Errori di confine temporale.** CNN1D mostra un pattern simile al BiLSTM (anticipo inizio, ritardo fine) ma con entità ridotta: −4.1 frame all'inizio e +5.1 alla fine. La mediana dell'inizio è −1.0 (non zero come negli altri modelli), indicando un bias sistematico lieve ma presente anche nelle predizioni tipiche.

| | Media | Std | Mediana |
|---|---|---|---|
| Errore inizio (frame) | −4.1 | 55.5 | −1.0 |
| Errore fine (frame) | +5.1 | 55.8 | 0.0 |

| Classe | Errore medio (frame) | Segmenti |
|---|---|---|
| Wash strainer | 167.9 | 8 |
| Cut olive | 156.6 | 12 |
| Move Around bacon | 126.9 | 41 |
| Wash bowl | 113.2 | 11 |
| Cut tomato | 105.9 | 65 |

**Confusioni tra classi.** Il dato più rilevante per CNN1D è la **magnitudine** delle confusioni: "Mix mixture,eating_utensil" → "Mix egg" conta 781 frame errati, contro i 134 frame del caso peggiore in MS-TCN++. Questo riflette direttamente il limite del campo recettivo di ~9 frame (≈0.4 s): una volta che il modello classifica erroneamente un'azione, non ha contesto temporale sufficiente per correggersi e mantiene la predizione sbagliata per centinaia di frame consecutivi.

| GT | Predetto | Frame |
|---|---|---|
| Mix mixture,eating_utensil | Mix egg | 781 |
| Wash pan | Wash eating_utensil | 307 |
| Wash pot | Wash pan | 225 |
| Put sponge | Take sponge | 161 |
| Cut tomato | Cut carrot | 154 |

Le coppie confuse seguono un pattern verbo-simile/oggetto-diverso (Mix→Mix, Wash→Wash, Cut→Cut), ma a differenza di MS-TCN++ e LSTM le confusioni persistono su segmenti molto più lunghi — evidenza che senza memoria temporale il modello non riesce a raccogliere abbastanza contesto per disambiguare azioni visivamente simili.

### 5.4 Analisi qualitativa — xLSTM

L'analisi è condotta sul checkpoint migliore di fold 1 (`fold1-epoch=91-val/edit_score`), valutato su split 3 con sliding window (stride=64) su 2021 clip.

**Errori di confine temporale.** xLSTM è il modello **più preciso sui confini** tra quelli testati: anticipa l'inizio di soli 2.6 frame e ritarda la fine di 1.0 frame, con una deviazione standard sulla fine (11.6 frame) molto più bassa di MS-TCN++ (34.5) e LSTM (58.3). Questo è coerente con il boundary F1 più alto della Tabella 1 (85.3). La mediana nulla su entrambi gli estremi conferma che la maggioranza delle predizioni cade esattamente sul confine GT.

| | Media | Std | Mediana |
|---|---|---|---|
| Errore inizio (frame) | −2.6 | 29.4 | 0.0 |
| Errore fine (frame) | +1.0 | 11.6 | 0.0 |

| Classe | Errore medio (frame) | Segmenti |
|---|---|---|
| Move Around patty | 53.7 | 31 |
| Move Around bacon | 45.4 | 41 |
| Cut onion | 27.6 | 44 |
| Cut carrot | 25.1 | 64 |
| Cut bell_pepper | 16.1 | 41 |

**Confusioni tra classi.** La magnitudine delle confusioni è nettamente inferiore agli altri modelli: il caso peggiore conta 17 frame, contro i 48 di MS-TCN++ e i 781 di CNN1D. Questo riflette la segmentazione più pulita del modello, capace di sfruttare il memory mixing per mantenere predizioni coerenti su sequenze lunghe senza frammentazione.

| GT | Predetto | Frame |
|---|---|---|
| Put cucumber | Take cucumber | 17 |
| Take plate | Put eating_utensil | 7 |
| Take bell_pepper | Cut bell_pepper | 6 |
| Operate stove | Move Around bacon | 5 |
| Take condiment_container | Move Around patty | 5 |

Il pattern di confusione è misto (Put↔Take sullo stesso oggetto, Take↔Cut sullo stesso oggetto): combina sia l'errore di verbo dell'LSTM sia l'errore di oggetto di MS-TCN++, ma su un numero di frame troppo piccolo per costituire un errore sistematico rilevante.

### 5.5 Extra Objective — Soft-NMS Post-Processing

**Motivazione.** Soft-NMS (*He et al., 2017*) è un'alternativa al NMS standard per sopprimere detection ridondanti: invece di eliminare le bounding box che sovrappongono una detection più confidente, ne riduce lo score con un decay gaussiano proporzionale alla IoU:

```
score_j  ←  score_j · exp( − IoU(i, j)² / σ )
```

Le box con score sotto una soglia vengono poi scartate. L'obiettivo applicato alla segmentazione temporale è sopprimere segmenti brevi e poco affidabili che frammentano predizioni di azioni lunghe.

**Implementazione.** Le funzioni `soft_nms_proposals()`, `_extract_proposals()` e `_reconstruct_dense()` in `src/evaluation/evaluate.py` implementano l'algoritmo adattato alla segmentazione con sliding window:

1. Per ogni finestra di inferenza (128 frame, stride 64 → 50% overlap), si estraggono segmenti contigui `[start_abs, end_abs, class, score]` con coordinate assolute nel clip. Lo score è la probabilità softmax media della classe predetta sulla finestra.
2. Tutte le proposte di tutte le finestre vengono raccolte in una lista unica. Le proposte provenienti da finestre sovrapposte **si sovrappongono per costruzione** (stesse coordinate assolute), il che permette a Soft-NMS di calcolare IoU > 0.
3. Le proposte sono ordinate per score decrescente. Per ogni coppia same-class con IoU > 0 si applica il decay: `score_j *= exp(−IoU(i,j)² / σ)`.
4. Le proposte con score sotto `score_thresh` vengono scartate.
5. La sequenza densa viene ricostruita sovrascrivendo in ordine di score crescente (i più forti per ultimi).

Il flag `--soft-nms` attiva il post-processing; `--soft-nms-sigma` (default 0.5) e `--soft-nms-thresh` (default 0.01) controllano aggressività del decay e soglia di soppressione.

Il vantaggio rispetto ad applicare Soft-NMS sull'argmax finale è che le finestre sovrapposte producono proposte con coordinate assolute che si sovrappongono, abilitando IoU > 0:

```
Clip:    |-------- Cut tomato --------|---- background ----|
         0        20       50         80                  120

Win 1:   |-------- 128 frame ----------|
          → proposta: [20, 50, "Cut tomato", score=0.91]

Win 2:           |-------- 128 frame ----------|   (offset +64)
          → proposta: [20, 50, "Cut tomato", score=0.85]

IoU(prop1, prop2) = 31/31 = 1.0
decay:  score_2 *= exp(−1.0²/0.5) = 0.85 × 0.135 = 0.115  → soppressa
```

Senza questo adattamento (Soft-NMS sull'argmax denso), i segmenti sarebbero sempre disgiunti e il decay non scatterebbe mai.

**Risultati comparativi — fold 1, split 3, 2021 clip (σ=0.5, thr=0.01).**

| Modello | Err. inizio senza | Err. inizio con | Err. fine senza | Err. fine con | Top confusione (frame) senza → con |
|---|:---:|:---:|:---:|:---:|---|
| MS-TCN++ | −9.7 | −9.4 | +10.4 | +10.3 | Put→Cut cucumber: 138 → 133 |
| LSTM     | −8.6 | −8.6 | +9.2  | +9.0  | Put→Cut cucumber: 37 → 36   |
| CNN1D    | −4.1 | −4.1 | +5.1  | +4.9  | Mix mixture→Mix egg: 781 → 780 |

Soft-NMS produce effetti reali ma contenuti su tutti e tre i modelli. La riduzione più evidente è sull'errore di fine azione (−0.1 frame per tutti) e sulla top confusion di MS-TCN++ (−5 frame). L'effetto è limitato dalla coerenza delle finestre sovrapposte: se due finestre concordano già sulla stessa classe, la proposta più debole ha score comunque alto e il decay non la sopprime. Il beneficio maggiore si otterrebbe su modelli meno precisi o con σ più piccolo (decay più aggressivo).

**Nota sull'adattamento.** Applicare Soft-NMS **dopo** la media dei logit e l'argmax non funziona: i segmenti estratti da una predizione densa sono non sovrapposti per costruzione (IoU = 0 sempre), quindi il decay non scatterebbe mai. La chiave è operare **prima** della media, raccogliendo proposte da ogni finestra di inferenza. Le finestre con stride < seq_len producono proposte sovrapposte, rendendo l'algoritmo applicabile esattamente come nel dominio detection.

---

## 6. Conclusion and Limitations

Il progetto ha costruito una pipeline completa e riproducibile per la segmentazione temporale delle azioni su EGTEA Gaze+, con i seguenti contributi metodologici rispetto allo stato iniziale:

- **Protocollo di valutazione**: la pipeline implementa il **protocollo ufficiale EGTEA a 3 fold** sugli split ufficiali (`train.py`), superando il subset casuale del solo split 1; i risultati riportati in questo report si riferiscono al fold 1.
- **Architettura MS-TCN++**: allineata al repository ufficiale MS-TCN2 con Prediction_Generation (doppio flusso dilated) per il primo stage e Refinement stages successivi.
- **Smooth loss**: allineata all'implementazione ufficiale (asimmetrica, senza maschera GT, λ_s=0.15).
- **Metriche**: allineate al file `metrics.py` ufficiale (frame accuracy su tutti i frame, edit score 0–100, F1@{10,25,50} segment-level, boundary F1); mIoU rimosso.
- **Pipeline DINOv3**: estrazione streaming di feature ViT-B da video raw, alternativa alle feature TSN pre-estratte.

**Risultati principali (fold 1, feature DINOv3 ViT-B).** Le quattro architetture si separano in tre fasce (Tabella 1, §5). CNN1D resta la baseline più debole (edit 73.5), penalizzata dal campo recettivo limitato. LSTM e MS-TCN++ sono appaiati in testa sulle metriche di segmento (edit 94.8 e 94.5, F1@10 ~88.7), grazie al contesto temporale ampio. xLSTM ottiene la migliore frame accuracy (98.0) e il miglior boundary F1 (85.3), eccellendo nella precisione di confine ma risultando meno solido sulla struttura sequenziale (edit 91.4). Nessun modello domina su tutte le metriche: la scelta dipende dall'obiettivo (struttura sequenziale → LSTM/MS-TCN++; precisione di confine → xLSTM).


---

## 7. Additional Information

### 7.1 Contribution Breakdown

- **Simone Battiato**: progettazione dell'intera pipeline, implementazione di tutti i modelli (CNN1D, LSTM, xLSTM, MS-TCN++), loss functions, metriche di valutazione, script di analisi e valutazione qualitativa.
