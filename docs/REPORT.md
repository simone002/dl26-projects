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
- **Mamba** — State Space Model (SSM) con selective state spaces; usa la libreria ufficiale `mamba-ssm`.
- **MS-TCN++** — Multi-Stage TCN con convoluzioni dilatate a crescita esponenziale su 4 stage; ogni stage raffina le predizioni dello stage precedente.

Rispetto al semplice uso di codice esistente, i contributi tecnici principali sono:
- Loss combinata CE + Smooth + Boundary con motivazione esplicita per ciascun termine.
- Sliding window in validazione con overlap 50% per una valutazione riproducibile.
- Data augmentation temporale (shift casuale delle label) per ridurre il bias di anticipazione/ritardo.
- **Pipeline di estrazione feature DINOv3**: script di estrazione streaming (`src/utils/extract_dinov3_features.py`) che processa i video raw EGTEA frame per frame con DINOv3 ViT-B (`facebook/dinov3-vitb16-pretrain-lvd1689m`), producendo vettori d = 768 per frame salvati come file `.npy` memory-mapped.
---

## 3. Data Used

### Dataset: EGTEA Gaze+

EGTEA Gaze+ (Li et al., *ECCV* 2018) è un dataset egocentric acquisito con una telecamera montata sulla testa di 32 soggetti durante sessioni di preparazione pasti in cucina. Il dataset comprende 86 sessioni video per un totale di circa 28 ore a 24 fps.

Le azioni sono annotate con **106 classi** composte da combinazioni verbo-oggetto (es. *Cut tomato*, *Mix egg*, *Pour water*) più una classe background. Sono presenti circa 10.325 istanze di azione distribuite su 7 tipi di pasto.

### Analisi del dataset (split 1, 8.299 clip)

L'analisi statistica su split 1 rivela tre caratteristiche strutturali che motivano le scelte progettuali della pipeline.

**Dominanza del background.** Il background occupa in media il **56%** dei frame per clip; 532 clip (6.4%) sono interamente background e non contengono alcun frame di azione. Questa proporzione giustifica la riduzione del peso del background a `bg_weight = 0.05` nella cross-entropy: senza questo accorgimento, un modello che predice sempre background otterrebbe un'accuracy del 56% senza apprendere nulla sulle 106 classi di interesse.

**Clip prevalentemente brevi.** La durata mediana è **59 frame** (~2.5 s a 24 fps) e il 75° percentile è 103 frame. Con `seq_len = 128`, l'80.8% dei clip viene coperto interamente in una singola finestra; il rimanente 19.2% richiede la sliding window in inferenza. Portare `seq_len = 256` porterebbe la copertura al **92.3%**, ma raddoppia la memoria e il costo computazionale per batch — si è preferito 128. La coda lunga (max 2.801 frame) indica che alcune sessioni sono molto più lunghe della norma.

**Sbilanciamento tra classi foreground.** La distribuzione delle classi segue una coda lunga marcata: la classe più frequente (*Open fridge*, 19.970 frame) supera di **20×** le classi nella metà inferiore della distribuzione. Le classi *Cut tomato*, *Cut cucumber*, *Cut onion* e *Cut carrot* sono tra le più frequenti e visivamente simili tra loro — fatto che spiega la concentrazione di confusioni su queste coppie nell'analisi qualitativa (§5.1).

### Protocollo di valutazione: 3-fold cross-validation

Viene usato il **protocollo ufficiale EGTEA** a 3 fold, che sfrutta tutti e tre gli split ufficiali del dataset:

| Fold | Train | Val / Test |
|---|---|---|
| 1 | split 1 + split 2 | split 3 |
| 2 | split 1 + split 3 | split 2 |
| 3 | split 2 + split 3 | split 1 |

Ogni fold utilizza circa **16.600 clip di training** (due split concatenati via `ConcatDataset`). Il set di test è ricavato dall'80% dei clip del test split (circa **2.700 finestre**), tenendo il restante 20% (~670 finestre) come test finale indipendente. I risultati finali sono la media e la deviazione standard delle metriche sui 3 fold.

### Feature

Le feature sono estratte dai video raw con **DINOv3 ViT-B** (Siméoni et al., *arXiv* 2025; `facebook/dinov3-vitb16-pretrain-lvd1689m`), producendo vettori **d = 768** per ogni frame. L'estrazione usa lo script `src/utils/extract_dinov3_features.py` in modalità streaming (un batch di frame alla volta) e salva un file `.npy` per video. Il backbone è pre-addestrato su 1.6B immagini (LVD-1689M), potenzialmente più discriminativo per le feature visive fine-grained del dataset rispetto a backbone task-specific. I modelli non elaborano mai i pixel grezzi.

### Preprocessing e augmentation

**Training — random crop**: ad ogni epoca viene estratta una finestra casuale di `seq_len = 128` frame da ciascun clip, con:
- **Gaussian noise sulle feature** (`std = 0.05`): le feature DINOv3 sono estratte una volta e fisse. Il modello, vedendo sempre gli stessi vettori per gli stessi frame, rischierebbe di memorizzarli invece di imparare pattern generalizzabili. Il rumore simula il fatto che la stessa azione filmata in condizioni leggermente diverse produrrebbe feature leggermente diverse, rendendo il modello robusto a piccole perturbazioni della rappresentazione visiva.
- **Feature dropout casuale** (`prob = 0.1`): azzera casualmente il 10% dei vettori di feature per frame. Forza il modello a non affidarsi a singoli frame per prendere decisioni, incoraggiando l'uso del contesto temporale circostante.
- **Shift temporale casuale delle label** (`±5 frame`, `prob = 0.5`): sposta le annotazioni GT di un offset casuale. Poiché i confini tra azioni sono soggettivi e spesso incerti di qualche frame, questo previene l'overfitting sulla posizione esatta dei confini annotati e riduce il bias di anticipazione/ritardo nella predizione.

**Validation/Test — sliding window**: l'intera durata del clip viene coperta con finestre di 128 frame e `stride = 64` (overlap 50%), così ogni frame viene valutato.

```
Clip:  |-----------------------------------------------|
       seq_len=128          stride=64

Win 1: |-----128-----|
Win 2:         |-----128-----|
Win 3:                 |-----128-----|
```

**Perché la sliding window in validazione e non in training?** In training si usa un random crop perché l'obiettivo è l'augmentation: estrarre finestre casuali da posizioni diverse ad ogni epoca espone il modello a contesti variabili e riduce l'overfitting su sequenze specifiche. In validazione e test, invece, il requisito è la **copertura deterministica e completa** del clip: ogni frame deve essere valutato esattamente nella stessa condizione ad ogni run per garantire la riproducibilità delle metriche.

**Perché non padding o truncation?** La durata dei clip in EGTEA varia da pochi frame fino a 2.801 frame (75° percentile: 103 frame, massimo: 2.801 frame). Un padding al massimo sarebbe proibitivo in termini di memoria e computazione; una truncation a 128 frame escluderebbe completamente il 19.2% dei clip che eccede quella lunghezza, introducendo un bias sistematico verso i clip brevi.

**Perché stride = 64 (overlap 50%)?** Con overlap del 50% ogni frame interno al clip cade in almeno due finestre diverse. Le predizioni delle finestre sovrapposte vengono poi aggregate per maggioranza (argmax sui logit medi): i frame ai bordi di ogni finestra, dove il contesto disponibile è ridotto, vengono corretti dalla finestra adiacente che li vede invece come frame centrali, con contesto completo da entrambi i lati. Questo riduce sistematicamente gli artefatti ai confini delle finestre. Uno stride maggiore (es. 128, nessun overlap) lascerebbe i frame di giuntura senza correzione; uno stride minore (es. 32) migliorerebbe la robustezza ai bordi ma triplica il numero di passate di inferenza senza un guadagno proporzionale sulle metriche.

---

## 4. Methodology and Architecture

### 4.1 Architetture

Tutti i modelli condividono l'interfaccia `Input: (B, T, feat_dim)` → `Output: (B, T, num_classes)`.

**CNN1D** (`src/models/cnn1d.py`) — tratta la sequenza temporale come un segnale 1D e applica convoluzioni su di essa. Con `kernel_size=3` ogni frame vede solo sé stesso e i 2 vicini; stacking 4 layer porta il campo recettivo a 9 frame (~0.4 s a 24 fps). La **residual connection** (`x + conv(x)`) evita la scomparsa del gradiente nei layer profondi: il layer impara solo la correzione rispetto all'input, non l'intera trasformazione.

Limite principale: 9 frame di contesto sono insufficienti per azioni che durano secondi. Il modello non ha nessun meccanismo per collegare frame lontani, il che spiega i lunghi tratti di confusione osservati nell'analisi qualitativa (§5.3).

**LSTM** (`src/models/lstm.py`) — (Hochreiter & Schmidhuber, *Neural Computation* 1997) processa la sequenza frame per frame mantenendo uno stato nascosto che accumula informazioni su tutto ciò che ha visto. La variante **bidirezionale** esegue due passate:

```
f1 → f2 → f3 → ... → fT   (forward:  vede il passato)
fT → ...→ f3 → f2 → f1    (backward: vede il futuro)
                ↓
    concatenazione → proiezione lineare → logit
```

Ogni frame viene predetto con contesto sia passato che futuro (`hidden=256`, bidirezionale → 512 dim → Linear(512, 106 classi)). Limite: con sequenze lunghe l'LSTM tende a dimenticare eventi lontani nonostante i gate, e il loop temporale è sequenziale, rendendo il training più lento delle architetture convoluzionali.

**xLSTM** (`src/models/xlstm.py`) — (Beck et al., *NeurIPS* 2024) versione potenziata dell'LSTM in cui la memoria non è un vettore ma una **matrice H×H**, aggiornata tramite proiezioni Q, K, V (mLSTM con multi-head attention). Usa la libreria ufficiale NX-AI/xlstm. La maggiore capacità espressiva comporta un rischio più elevato di overfitting su dataset di medie dimensioni.

**Mamba** (`src/models/mamba.py`) — (Gu & Dao, *ICLR* 2024) architettura basata su **State Space Models** (SSM): modella la sequenza come un sistema dinamico che mappa input x(t) in output y(t) attraverso uno stato nascosto h(t):

```
h'(t) = A·h(t) + B·x(t)       (continuo)
h_t   = Ā·h_{t-1} + B̄·x_t   (discretizzato, passo Δ ZOH)
y_t   = C·h_t
```

L'innovazione chiave è la **selective scan**: le matrici B, C e il passo Δ dipendono dall'input corrente x_t (negli SSM classici *Linear Time Invariant* sono costanti). Il modello impara a selezionare quali informazioni incorporare o dimenticare nello stato nascosto — effetto analogo al gating dell'LSTM ma con complessità O(T) invece di O(T²) dell'attention.

Ogni `MambaBlock` applica: LayerNorm → SSM (con proiezione di espansione ×`expand`=2) → Dropout. Con `n_layers=8`, `hidden=256`, `d_state=16`, `d_conv=4`, `expand=2`, la proiezione interna lavora a 256×2=512 dimensioni pur mantenendo l'interfaccia a 256. Rispetto all'LSTM bidirezionale, Mamba opera in modo causale (forward only): non ha accesso al futuro, ma la selective scan con stato `d_state=16` permette di mantenere un contesto efficace su centinaia di frame. Si usa la libreria ufficiale `mamba-ssm`.

**MS-TCN++** (`src/models/mstcn.py`) — (Li et al., *IEEE TPAMI* 2020) — usa più stage successivi dove ognuno corregge gli errori del precedente, con architettura asimmetrica:

- **Stage 1 — Prediction_Generation**: doppio flusso di convoluzioni dilatate in parallelo sulla stessa sequenza:
  - *Stream 1*: dilation decrescente (2^(N−1) → 2^0)
  - *Stream 2*: dilation crescente (2^0 → 2^(N−1))
  - I due flussi sono fusi tramite `conv_fusion` (Conv1d(2·hidden, hidden, 1)), poi ReLU + dropout + connessione residuale.
- **Stage 2–4 — Refinement**: convoluzioni dilatate (2^0 → 2^(N−1)) con skip connections; ogni stage riceve il **softmax** dell'output del precedente. Se stage 1 è incerto tra *Cut tomato* (40%) e *Cut onion* (35%), stage 2 vede questa incertezza e usa il contesto temporale per risolverla.

Con 10 layer e dilation fino a 2^9=512 il campo recettivo copre l'intera finestra di 128 frame — come LSTM ma completamente parallelizzabile, poiché le convoluzioni non hanno dipendenze sequenziali.

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

Tutti i modelli convergono attorno a **20k–25k global step** (fold 1), corrispondenti tipicamente a ≈ 85–95 epoche prima che l'early stopping si attivi con patience 20 su `val/edit_score`.

---

## 5. Results and Discussion

Le metriche sono allineate all'implementazione ufficiale MS-TCN2: frame accuracy su tutti i frame, edit score (0–100), F1@{10,25,50} segment-level, boundary F1.

**Tabella 1**: Risultati 3-fold cross-validation — media ± std sui 3 fold — feature DINOv3 ViT-B.

| Model    | Acc (%)      | Edit Score   | F1@10        | F1@25        | F1@50        | Boundary F1  |
|----------|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| CNN1D    | 92.5 ± 0.7   | 72.9 ± 2.4   | 74.4 ± 1.8   | 73.8 ± 1.8   | 72.1 ± 1.6   | 57.1 ± 1.3   |
| LSTM     | 96.7 ± 0.3   | **95.4 ± 0.6** | 89.9 ± 0.8 | 89.7 ± 1.0   | 88.9 ± 1.0   | 78.2 ± 2.5   |
| xLSTM    | **98.1 ± 0.3** | 91.0 ± 0.3 | 86.8 ± 0.8   | 86.7 ± 0.8   | 86.6 ± 0.7   | **86.3 ± 1.0** |
| Mamba    | 97.9 ± 0.4   | 91.7 ± 0.4   | 87.5 ± 0.8   | 87.4 ± 0.9   | 87.2 ± 0.9   | 85.2 ± 0.5   |
| MS-TCN++ | 96.2 ± 0.4   | 94.9 ± 0.8   | 89.3 ± 1.2   | 89.1 ± 1.3   | 88.2 ± 1.4   | 78.0 ± 1.0   |

**Discussione comparativa.** I cinque modelli si separano in tre fasce nette, confermate dalla bassa deviazione standard su tutti i fold (≤ 2.5%).

- **CNN1D** è il fanalino di coda su tutte le metriche (edit 72.9 ± 2.4, boundary F1 57.1 ± 1.3). Il campo recettivo di ~9 frame non basta a modellare le dipendenze temporali: senza memoria, gli errori di classificazione persistono per centinaia di frame (§5.3), facendo crollare sia l'ordine dei segmenti (edit) sia la localizzazione dei confini.

- **LSTM e MS-TCN++** sono sostanzialmente appaiati in testa sulle metriche di segmento (edit 95.4 vs 94.9, F1@10 89.9 vs 89.3, F1@50 88.9 vs 88.2). LSTM supera di mezzo punto percentuale MS-TCN++ sull'edit score, con varianza minore (±0.6 vs ±0.8). Entrambi sfruttano un contesto temporale ampio — bidirezionale per l'LSTM, dilatato multi-stage per MS-TCN++ — e mostrano lo stesso comportamento ai confini (segmenti predetti più lunghi del GT, §5.1–5.2). Sono le due architetture di riferimento per questo task.

- **xLSTM e Mamba** hanno un profilo distinto e simile tra loro: entrambi vincono sulla **frame accuracy** (xLSTM 98.1 ± 0.3, Mamba 97.9 ± 0.4) e sul **boundary F1** (xLSTM 86.3 ± 1.0, Mamba 85.2 ± 0.5, contro ~78 di LSTM/MS-TCN++), ma restano indietro su edit (xLSTM 91.0, Mamba 91.7) e F1@k rispetto a LSTM/MS-TCN++. La lettura è che entrambi i modelli sono i più precisi a livello di singolo frame e di confine — predizione quasi esatta dei bordi — ma commettono più errori nella *struttura sequenziale* dei segmenti, dove edit ed F1@k li penalizzano. Mamba ha edit leggermente migliore di xLSTM (91.7 vs 91.0) e F1@k superiore, collocandosi a metà strada tra xLSTM e LSTM/MS-TCN++ sulla dimensione della coerenza sequenziale.

In sintesi, nessun modello domina su tutte le metriche: la scelta dipende dall'obiettivo (struttura sequenziale → LSTM/MS-TCN++; precisione di confine → xLSTM/Mamba), mentre la necessità di memoria temporale a lungo raggio è confermata dal divario netto rispetto alla baseline CNN1D.

**Dinamiche di training (fold 1).** L'analisi delle curve di training e validazione aggiunge contesto ai numeri finali. CNN1D raggiunge un plateau stabile attorno a **F1@10 ≈ 0.72** già nelle prime epoche, mentre tutti gli altri modelli convergono a **≈ 0.85** — un gap che si apre rapidamente e non si chiude, confermando che il limite del campo recettivo di 9 frame non è una questione di ottimizzazione ma di capacità strutturale. MS-TCN++ mostra un edit score più basso degli altri modelli *durante* il training (non solo nei numeri finali), con una traiettoria di convergenza più lenta rispetto a LSTM e xLSTM; questo suggerisce che il meccanismo multi-stage di refinement richiede più iterazioni per stabilizzarsi. Mamba, sorprendentemente, mostra F1 più basso di CNN1D, xLSTM e LSTM nelle prime fasi del training prima di recuperare verso i valori finali — traiettoria coerente con l'ottimizzazione degli SSM, che costruiscono lo stato selettivo in modo progressivo e non beneficiano delle stesse inizializzazioni favorevoli delle reti ricorrenti classiche.

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

### 5.5 Analisi qualitativa — Mamba

L'analisi è condotta sul checkpoint migliore di fold 1 (`fold1-epoch=98-val/edit_score=0.9`), valutato su split 3 con sliding window (stride=64) su 2021 clip.

**Errori di confine temporale.** Mamba ha un profilo molto simile a xLSTM: anticipa l'inizio di 2.4 frame e ritarda la fine di 2.0 frame, con deviazione standard (25.0 / 22.9 frame) analoga a xLSTM (29.4 / 11.6) e nettamente inferiore a MS-TCN++ e LSTM (~57–64 frame). La mediana nulla su entrambi gli estremi conferma che la maggioranza delle predizioni cade esattamente sul confine GT.

| | Media | Std | Mediana |
|---|---|---|---|
| Errore inizio (frame) | −2.4 | 25.0 | 0.0 |
| Errore fine (frame) | +2.0 | 22.9 | 0.0 |

| Classe | Errore medio (frame) | Segmenti |
|---|---|---|
| Wash bowl | 80.7 | 11 |
| Move Around bacon | 75.7 | 41 |
| Cut onion | 40.4 | 43 |
| Spread condiment,bread,eating_utensil | 27.3 | 32 |
| Move Around patty | 24.0 | 31 |

**Confusioni tra classi.** La magnitudine è la più bassa tra tutti i modelli: il caso peggiore conta 12 frame, contro i 17 di xLSTM, i 134 di MS-TCN++ e i 781 di CNN1D. Il pattern è coerente con xLSTM: confusioni miste verbo-oggetto su un numero di frame troppo piccolo per costituire un errore sistematico rilevante.

| GT | Predetto | Frame |
|---|---|---|
| Take plate | Put eating_utensil | 12 |
| Wash pan | Wash pot | 11 |
| Wash hand | Put cooking_utensil | 9 |
| Operate stove | Move Around bacon | 8 |
| Take bell_pepper | Cut bell_pepper | 8 |

Mamba conferma il profilo del selective state space model: boundary precision paragonabile a xLSTM e confusioni ancora più contenute, con la selective scan che enfatizza solo i frame informativi mantenendo la coerenza temporale su sequenze lunghe.

---

## 6. Conclusion and Limitations

Il progetto ha costruito una pipeline completa e riproducibile per la segmentazione temporale delle azioni su EGTEA Gaze+, con i seguenti contributi metodologici rispetto allo stato iniziale:

- **Protocollo di valutazione**: la pipeline implementa il **protocollo ufficiale EGTEA a 3 fold** sugli split ufficiali (`train.py`); i risultati riportati in questo report sono la media ± std sui 3 fold.
- **Architettura MS-TCN++**: allineata al repository ufficiale MS-TCN2 con Prediction_Generation (doppio flusso dilated) per il primo stage e Refinement stages successivi.
- **Smooth loss**: allineata all'implementazione ufficiale (asimmetrica, senza maschera GT, λ_s=0.15).
- **Metriche**: allineate al file `metrics.py` ufficiale (frame accuracy su tutti i frame, edit score 0–100, F1@{10,25,50} segment-level, boundary F1); mIoU rimosso.
- **Pipeline DINOv3**: estrazione streaming di feature ViT-B da video raw con salvataggio per-video in formato `.npy` memory-mapped.

**Risultati principali (3-fold CV, feature DINOv3 ViT-B).** Le cinque architetture si separano in tre fasce (Tabella 1, §5). CNN1D resta la baseline più debole (edit 72.9 ± 2.4), penalizzata dal campo recettivo limitato. LSTM e MS-TCN++ sono appaiati in testa sulle metriche di segmento (edit 95.4 ± 0.6 e 94.9 ± 0.8, F1@10 ~89.6), grazie al contesto temporale ampio. xLSTM ottiene la migliore frame accuracy (98.1 ± 0.3) e il miglior boundary F1 (86.3 ± 1.0), eccellendo nella precisione di confine. Mamba condivide il profilo di xLSTM (acc 97.9 ± 0.4, BF1 85.2 ± 0.5) con edit leggermente migliore (91.7 ± 0.4), collocandosi tra il cluster SSM/xLSTM e quello ricorrente/convolutivo sulla dimensione della coerenza sequenziale. Nessun modello domina su tutte le metriche: la scelta dipende dall'obiettivo (struttura sequenziale → LSTM/MS-TCN++; precisione di confine → xLSTM/Mamba).


---

## 7. Additional Information

### 7.1 Contribution Breakdown

- **Simone Battiato**: progettazione dell'intera pipeline, implementazione di tutti i modelli (CNN1D, LSTM, xLSTM, MS-TCN++), loss functions, metriche di valutazione, script di analisi e valutazione qualitativa.
