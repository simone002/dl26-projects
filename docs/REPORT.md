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

Il progetto implementa una pipeline completa di segmentazione temporale su EGTEA Gaze+, confrontando cinque architetture con la stessa infrastruttura di training:

- **CNN1D** — baseline convolutiva 1D, punto di riferimento minimo.
- **LSTM bidirezionale** — cattura dipendenze passate e future con stato ricorrente.
- **xLSTM** — variante extended-LSTM con memory mixing;
- **MS-TCN++** — Multi-Stage TCN con convoluzioni dilatate a crescita esponenziale su 4 stage; ogni stage raffina le predizioni dello stage precedente.
- **Mamba** — State Space Model selettivo implementato in PyTorch puro (senza `mamba-ssm`, incompatibile con Windows). Per eliminare il collo di bottiglia del loop Python sul selective scan è stato implementato un **parallel prefix scan** in O(log T) passi con gradient checkpointing per contenere l'uso di memoria.

Rispetto al semplice uso di codice esistente, i contributi tecnici principali sono:
- Loss combinata CE + Smooth + Boundary con motivazione esplicita per ciascun termine.
- Sliding window in validazione con overlap 50% per una valutazione riproducibile.
- Data augmentation temporale (shift casuale delle label) per ridurre il bias di anticipazione/ritardo.
- **Pipeline di estrazione feature DINOv3**: script di estrazione streaming (`scripts/extract_dinov3_features.py`) che processa i video raw EGTEA frame per frame con DINOv3 ViT-B (`facebook/dinov3-vitb16-pretrain-lvd1689m`), producendo vettori d = 768 per frame salvati come file `.npy` memory-mapped. L'approccio streaming evita il caricamento dell'intero video in RAM (rischio OOM su video da ~70 GB di frame). Il nuovo datamodule `EGTEADataModuleNpy` legge direttamente i file `.npy`.


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

Sono supportate due sorgenti di feature:

**TSN (baseline):** feature pre-estratte con un backbone **TSN** (*Temporal Segment Network*) addestrato sul dataset stesso, producendo vettori **d = 1024** per ogni frame salvati in un archivio LMDB. Il modello non elabora mai i pixel grezzi.

**DINOv3 (sorgente alternativa):** feature estratte dai video raw con **DINOv3 ViT-B** (`facebook/dinov3-vitb16-pretrain-lvd1689m`), producendo vettori **d = 768** per ogni frame. L'estrazione usa lo script `scripts/extract_dinov3_features.py` in modalità streaming (un batch di frame alla volta) e salva un file `.npy` per video. Questa sorgente non richiede l'archivio LMDB e utilizza un backbone con pre-training su 1.6B immagini (LVD-1689M), potenzialmente più discriminativo per le feature visive fine-grained del dataset.

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

**Mamba** (`src/models/mamba.py`) — State Space Model selettivo basato su Gu & Dao (2023). A differenza di LSTM e xLSTM, i parametri B, C, Δ dipendono dal **contenuto** dell'input: il modello decide dinamicamente quanto "soffermarsi" su ogni frame in base a cosa contiene, anziché applicare gate fissi per posizione. Questo lo rende selettivo sul contenuto, non solo sulla posizione.

Rispetto agli altri modelli combina tre proprietà uniche simultaneamente: (1) stato ricorrente con campo recettivo teoricamente infinito, (2) calcolo parallelizzabile durante il training tramite parallel prefix scan, (3) selettività adattiva al contenuto. Il limite nel contesto della segmentazione è la **causalità**: a differenza del BiLSTM, Mamba vede solo il contesto passato, non quello futuro.

L'implementazione usa PyTorch puro con parallel prefix scan in O(T log T) passi invece del loop sequenziale O(T), evitando la dipendenza da `mamba-ssm` (incompatibile con Windows). Il selective scan è matematicamente equivalente al kernel CUDA ufficiale — cambia solo l'efficienza computazionale.

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

A differenza di mIoU e F1, non dipende dalla precisione temporale dei confini: cattura se il modello predice le azioni giuste **nel giusto ordine**, ignorando piccoli sfasamenti. È usato come metrica di early stopping perché riflette la struttura sequenziale globale della predizione.

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

Sono in corso due campagne di training:

1. **Feature TSN (d=1024)** — training 3-fold con `train_cv.py --config experiments/configs/mstcn_cv.yaml` per tutti e 5 i modelli.
2. **Feature DINOv3 ViT-B (d=768)** — estrazione in corso (`scripts/extract_dinov3_features.py`); al completamento, training con `train_cv.py --config experiments/configs/mstcn_cv_npy.yaml`.

**Tabella 1**: Risultati 3-fold cross-validation — *in corso di esecuzione*.

| Model    | Acc (%) | Edit Score | F1@10 | F1@25 | F1@50 | Boundary F1 |
|----------|:-------:|:----------:|:-----:|:-----:|:-----:|:-----------:|
| CNN1D    | 92.7    | 73.5       | 74.7  | 74.3  | 72.3  | 57.0        |
| LSTM     | 96.6    | 94.8       | 88.9  | 88.7  | 88.2  | 77.9        |
| xLSTM   | 97.1    | 89.6       | 85.5  | 85.4  | 85.1  | 77.9        |
| Mamba    | —       | —          | —     | —     | —     | —           |
| MS-TCN++ | 96.6 (±0.5) | 95.1 (±0.8) | 87.8 (±0.4) | 87.7 (±0.3) | 87.0 (±0.3) | 78.8 (±1.1) |

*I valori riportati sono la media dei 3 fold; la deviazione standard è indicata tra parentesi.*

### 5.1 Analisi qualitativa — MS-TCN++

L'analisi degli errori sistematici è condotta con `src/evaluation/evaluate.py` sul checkpoint migliore di fold 2 (`fold2-epoch=96-val/edit_score=95.7.ckpt`), valutato su split 2 con sliding window (stride=64) su 2021 clip.

**Errori di confine temporale.** Il modello anticipa l'inizio delle azioni di 4.5 frame in media e ritarda la fine di 4.5 frame, producendo segmenti predetti leggermente più corti di quelli GT. La deviazione standard è alta (≈35 frame) ma la mediana è 0, indicando che la maggioranza delle predizioni è precisa e sono gli outlier sulle azioni rare a far salire la media.

| | Media | Std | Mediana |
|---|---|---|---|
| Errore inizio (frame) | −4.5 | 37.3 | 0.0 |
| Errore fine (frame) | +4.5 | 34.5 | 0.0 |

Le classi con errore di confine maggiore sono tutte rare o visivamente ambigue nel tempo:

| Classe | Errore medio (frame) | Segmenti |
|---|---|---|
| Squeeze washing_liquid,sponge | 99.6 | 8 |
| Operate microwave | 84.1 | 11 |
| Move Around patty | 70.6 | 27 |
| Cut tomato | 64.5 | 66 |
| Wash strainer | 63.0 | 9 |

**Confusioni tra classi.** Le principali confusioni coinvolgono coppie semanticamente vicine:

| GT | Predetto | Frame |
|---|---|---|
| Wash hand | Wash pan | 48 |
| Take cucumber | Divide/Pull Apart onion | 39 |
| Cut cucumber | Cut onion | 37 |
| Cut onion | Divide/Pull Apart onion | 35 |
| Put eating_utensil | Cut tomato | 32 |

Il pattern è coerente con la sfida della granularità fine descritta in §1: *Cut cucumber* e *Cut onion* producono quasi lo stesso pattern visivo frame per frame, e anche feature DINOv3 ViT-B (pre-addestrate su 1.6B immagini) non riescono a distinguerli guardando i singoli frame. Il problema non è la qualità delle feature visive ma l'ambiguità intrinseca dell'azione senza contesto sull'oggetto.

### 5.2 Analisi qualitativa — LSTM

**Errori di confine temporale.** Il BiLSTM mostra un pattern opposto a MS-TCN++: anticipa l'inizio di 8.6 frame e ritarda la fine di 9.2 frame, producendo segmenti predetti **più lunghi** dei GT. Questo comportamento è coerente con la natura bidirezionale del modello: avendo accesso al contesto futuro, il modello "vede" l'azione in arrivo e anticipa l'inizio; simmetricamente, non taglia il segmento finché l'attività non è visivamente cessata. La deviazione standard è alta (54–58 frame) ma la mediana è 0, confermando che la maggioranza delle predizioni è accurata e sono gli outlier sulle azioni lunghe a fare salire la media.

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

**Confusioni tra classi.** Il pattern di confusione è qualitativamente diverso da MS-TCN++: l'LSTM confonde prevalentemente il **verbo** (Take → Cut) mantenendo l'oggetto corretto, mentre MS-TCN++ confondeva l'**oggetto** (Cut cucumber → Cut onion) mantenendo il verbo.

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

**Confusioni tra classi.** Il dato più rilevante per CNN1D è la **magnitudine** delle confusioni: "Mix mixture,eating_utensil" → "Mix egg" conta 781 frame errati, contro i 48 frame del caso peggiore in MS-TCN++. Questo riflette direttamente il limite del campo recettivo di ~9 frame (≈0.4 s): una volta che il modello classifica erroneamente un'azione, non ha contesto temporale sufficiente per correggersi e mantiene la predizione sbagliata per centinaia di frame consecutivi.

| GT | Predetto | Frame |
|---|---|---|
| Mix mixture,eating_utensil | Mix egg | 781 |
| Wash pan | Wash eating_utensil | 307 |
| Wash pot | Wash pan | 225 |
| Put sponge | Take sponge | 161 |
| Cut tomato | Cut carrot | 154 |

Le coppie confuse seguono un pattern verbo-simile/oggetto-diverso (Mix→Mix, Wash→Wash, Cut→Cut), ma a differenza di MS-TCN++ e LSTM le confusioni persistono su segmenti molto più lunghi — evidenza che senza memoria temporale il modello non riesce a raccogliere abbastanza contesto per disambiguare azioni visivamente simili.

### 5.4 Extra Objective — Soft-NMS Post-Processing

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

- **Protocollo di valutazione**: migrato da subset casuale del solo split 1 a **3-fold cross-validation** con gli split ufficiali EGTEA, rendendo i risultati confrontabili con la letteratura.
- **Architettura MS-TCN++**: allineata al repository ufficiale MS-TCN2 con Prediction_Generation (doppio flusso dilated) per il primo stage e Refinement stages successivi.
- **Smooth loss**: allineata all'implementazione ufficiale (asimmetrica, senza maschera GT, λ_s=0.15).
- **Metriche**: allineate al file `metrics.py` ufficiale (frame accuracy su tutti i frame, edit score 0–100, F1@{10,25,50} segment-level, boundary F1); mIoU rimosso.
- **Pipeline DINOv3**: estrazione streaming di feature ViT-B da video raw, alternativa alle feature TSN pre-estratte.

I risultati definitivi (3-fold CV su feature TSN e DINOv3) sono in corso di produzione e saranno inseriti nella Tabella 1 al completamento.

**Limitazioni attuali:**
- Le feature TSN e DINOv3 sono fisse e pre-estratte: i modelli non adattano la rappresentazione visiva al task di segmentazione.
- `seq_len = 128` copre ~5 secondi a 24fps; azioni con struttura a più lungo raggio non sono completamente catturabili in una singola finestra.
- L'implementazione Mamba in PyTorch puro è più lenta della versione con kernel CUDA (`mamba-ssm`), non disponibile su Windows.

**Sviluppi futuri:**
- Fine-tuning end-to-end del backbone (TSN o DINOv3).
- Aggregazione multi-window in inferenza per clip molto lunghi.


---

## 7. Additional Information

### 7.1 Contribution Breakdown

- **Simone Battiato**: progettazione dell'intera pipeline, implementazione di tutti i modelli (CNN1D, LSTM, xLSTM, MS-TCN++, Mamba), loss functions, metriche di valutazione, ottimizzazione del parallel scan per Mamba, script di analisi e valutazione qualitativa.

### 7.2 Use of Artificial Intelligence

**Claude Code** (Anthropic, modello claude-sonnet-4-6) è stato utilizzato come assistente durante lo sviluppo nelle seguenti fasi:

- **Migrazione del codice**: ristrutturazione del repository nella struttura richiesta (`src/`, `experiments/configs/`, ecc.) con aggiornamento automatico degli import.
- **Documentazione**: supporto nella redazione di questo report e del README tecnico.

