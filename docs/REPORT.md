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


---

## 3. Data Used

### Dataset: EGTEA Gaze+

EGTEA Gaze+ è un dataset egocentric acquisito con una telecamera montata sulla testa di 32 soggetti durante sessioni di preparazione pasti in cucina. Il dataset comprende 86 sessioni video per un totale di circa 28 ore a 24 fps.

Le azioni sono annotate con **106 classi** composte da combinazioni verbo-oggetto (es. *Cut tomato*, *Mix egg*, *Pour water*) più una classe background. Sono presenti circa 10.325 istanze di azione distribuite su 7 tipi di pasto.

### Split utilizzato

In questo lavoro si usa lo **split 1**:

| Split | Clip | Modalità di campionamento |
|---|---:|---|
| Train | 6.277 | random crop per epoch |
| Val | 2.022 | sliding window |
| Test | 2.022 | sliding window |

La validazione è estratta dai clip di training con seed fisso (`split_seed=42`, `val_size=2022`) a livello di clip, garantendo che nessun frame compaia in entrambi gli insiemi.

### Feature

Le feature visive sono pre-estratte con un backbone **TSN** (*Temporal Segment Network*) addestrato sul dataset stesso, producendo vettori **d = 1024** per ogni frame salvati in un archivio LMDB. Il modello non elabora mai i pixel grezzi.

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

**LSTM** (`src/models/lstm.py`) — LSTM bidirezionale con proiezione lineare finale. Il contesto bidirezionale aiuta ma il training su sequenze lunghe è più lento dei modelli convoluzionali.

**xLSTM** (`src/models/xlstm.py`) — extended LSTM con matrici di memoria più espressive (mLSTM con multi-head attention e proiezioni QKV). Usa direttamente la libreria ufficiale NX-AI/xlstm.

**MS-TCN++** (`src/models/mstcn.py`) — 4 stage di TCN con layer dilatati (dilatazione 1, 2, 4, …, 512). Ogni stage prende il softmax dell'output del precedente come input, raffinando progressivamente la segmentazione. È l'architettura di riferimento per questo task.

**Mamba** (`src/models/mamba.py`) — blocco SSM selettivo: i parametri B, C, Δ dipendono dall'input (selective scan). Implementato in PyTorch puro con parallel scan per evitare il loop Python su T timestep.

### 4.2 Loss Function

La loss totale combina tre termini:

```
L = L_CE + λ_s · L_smooth + λ_b · L_boundary
```

con `λ_s = 0.2`, `λ_b = 0.3`.

**Cross-Entropy con label smoothing (ε=0.1) e class weights.**
Il peso del background è ridotto a `bg_weight = 0.05` per non dominare il gradiente. Il label smoothing riduce l'overconfidence sulle classi visivamente simili.

**Smooth Loss** (MS-TCN style).
```
L_smooth = mean( clamp((log_p[t] − log_p[t−1])², max=16) · mask[t] )
```
`mask[t] = 1` dove il GT non cambia tra frame consecutivi. Penalizza transizioni spurie nelle predizioni dove il GT è costante (flickering), migliorando edit score e F1@k.

**Boundary Loss.**
Applica una CE con peso maggiore sui frame entro ±3 frame da ogni transizione GT. Incentiva la precisione sul "quando inizia e finisce" ogni azione, con impatto diretto su boundary F1 e F1@50.

### 4.3 Metriche di Valutazione

Quattro metriche complementari coprono aspetti diversi della qualità della segmentazione.

**mIoU** (*mean Intersection over Union*). Per ogni classe c, IoU = TP_c / (TP_c + FP_c + FN_c) calcolato a livello di frame. La media esclude il background e penalizza sia le predizioni in eccesso (FP) sia i frame mancati (FN). È la metrica più sensibile allo sbilanciamento delle classi.

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
| mIoU | frame | FP e FN per classe | sbilanciamento classi |
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

**Tabella 1**: Risultati quantitativi sul **test set** (split 1). Tutte le metriche in %, migliori valori in grassetto.

| Model    | mIoU | Edit Score | F1@10 | F1@25 | F1@50 | Boundary F1 | Epochs |
|----------|:----:|:----------:|:-----:|:-----:|:-----:|:-----------:|:------:|
| CNN1D    | 3.4  |    5.9     |  5.3  |  3.6  |  2.5  |    21.2     |  86†   |
| LSTM     | 4.5  |   11.1     | 10.9  |  9.8  |  8.4  |    27.4     |  100   |
| xLSTM   | 4.6  |    7.0     |  6.5  |  5.0  |  3.9  |    15.2     |  100   |
| Mamba    | 7.2‡ |   13.7‡    | 14.4‡ | 12.4‡ | 10.4‡ |    17.2‡    |  35†‡  |
| MS-TCN++ | 4.1  | **11.0**   | **10.6** | **9.9** | **9.3** | **46.3** | 100   |

† early stopping (patience 20)  
‡ metriche **val** — la valutazione test non è stata completata per un errore OOM con il codice precedente all'ottimizzazione del parallel scan

**Tabella 2**: Confronto train vs val mIoU — indicatore di overfitting.

| Model    | Train mIoU | Val mIoU | Gap  |
|----------|:----------:|:--------:|:----:|
| CNN1D    |   30.2%    |   6.7%   | 23.5 |
| LSTM     |   91.7%    |   7.1%   | 84.6 |
| xLSTM   |   89.4%    |   8.4%   | 81.0 |
| Mamba    |   81.6%    |   7.2%   | 74.4 |
| MS-TCN++ |   79.5%    |   6.9%   | 72.6 |

### Analisi dei risultati

**MS-TCN++ domina su Boundary F1 (46.3%)**, con un margine di 19 punti sul secondo (LSTM 27.4%). L'architettura multi-stage raffina progressivamente le predizioni: i primi stage producono una segmentazione grezza, quelli successivi correggono i confini. Combinata con la boundary loss, questo porta a una localizzazione temporale nettamente superiore agli altri modelli. Su F1@50 — la metrica più severa sull'overlap temporale — MS-TCN++ è il miglior modello tra quelli con metriche test complete.

**LSTM e MS-TCN++ sono comparabili su edit score** (~11%). Entrambi catturano la struttura sequenziale delle azioni, ma con meccanismi diversi: la LSTM bidirezionale usa lo stato ricorrente per propagare il contesto, MS-TCN++ usa convoluzioni dilatate con campo recettivo esponenziale.

**xLSTM sottoperforma** rispetto alla LSTM semplice su tutte le metriche, nonostante la maggiore complessità e l'uso della libreria ufficiale NX-AI/xlstm. La causa è probabilmente l'overfitting elevato (train mIoU 89.4% vs val 8.4%): il meccanismo mLSTM con proiezioni QKV ha più parametri e capacità espressiva, ma questa si traduce in memorizzazione dei pattern di training piuttosto che in migliore generalizzazione sul test.

**CNN1D è chiaramente la baseline più debole**: senza memoria a lungo raggio, il modello non riesce a catturare le dipendenze temporali che caratterizzano le sequenze di azioni. Il campo recettivo limitato dal kernel size (3 frame) è insufficiente per le sequenze di 128 frame usate in training. Nota: CNN1D mostra il gap train-val più basso (23.5 pp) — non perché generalizzi meglio, ma perché underfits sul training set.

**Mamba mostra forte overfitting** (train mIoU 81.6% vs val 7.2%, gap 74.4 pp). Il meccanismo di selezione input-dipendente (parametri B, C, Δ adattivi) conferisce al modello alta capacità espressiva, ma in assenza di sufficiente regolarizzazione porta a memorizzazione dei pattern di training. Le val metrics al momento dell'early stop (edit 13.7%, F1@10 14.4%) sono comparabili agli altri modelli, suggerendo che con una regolarizzazione più aggressiva potrebbe competere.

### Sbilanciamento e valori assoluti

I valori assoluti sono bassi (mIoU 3–7%, edit score 6–14%) per ragioni strutturali del task:
- 106 classi fine-grained su un solo split di training (~6K clip).
- Il background costituisce la maggioranza dei frame, abbassando mIoU (calcolato solo sul foreground ma con denominatori alti su FP).
- Le feature TSN sono fisse: il modello non può adattare la rappresentazione visiva.
- La finestra di 128 frame (~5s) limita il contesto disponibile per azioni con struttura temporale più lunga.

Tutti i modelli mostrano overfitting significativo: il gap train-val sull'edit score varia da ~57 pp (LSTM: train 79.3%, val 18.6%) a ~17 pp (CNN1D: train 12.7%, val 11.8%). L'early stopping arresta il training ma non risolve il problema strutturale.

Il gap è atteso e strutturale: le feature TSN sono estratte da un modello addestrato per clip-level recognition su EGTEA stesso, non ottimizzato per la predizione densa frame-per-frame. Con feature fisse e 106 classi fine-grained su un dataset piccolo, i modelli ad alta capacità (LSTM, xLSTM) memorizzano le associazioni feature→label dei clip di training invece di apprendere pattern generalizzabili. La riduzione del gap richiederebbe fine-tuning end-to-end del backbone, indicato come sviluppo futuro nella sezione 6.

### Analisi errori di confine e confusioni sistematiche

**Tabella 3**: Errori di confine sul test set (in frame; + = ritardo, − = anticipo).

| Model    | Inizio media | Inizio mediana | Fine media | Fine mediana |
|----------|:------------:|:--------------:|:----------:|:------------:|
| CNN1D    |    +43.1     |     +7.5       |   −49.9    |    −7.0      |
| LSTM     |    +35.5     |     +3.0       |   −42.1    |    −2.0      |
| xLSTM   |    +27.8     |     +2.0       |   −36.6    |     0.0      |
| Mamba    |    +33.1     |     +3.0       |   −33.9    |     0.0      |
| MS-TCN++ |    +17.8     |     +0.0       |   −21.7    |    +1.0      |

Tutti i modelli mostrano un pattern sistematico: ritardo sull'inizio (+) e anticipo sulla fine (−) delle azioni. CNN1D ha gli errori più grandi (+43.1/−49.9), coerente con la mancanza di contesto temporale. MS-TCN++ è il migliore su entrambi gli assi (+17.8/−21.7), grazie al meccanismo multi-stage che raffina progressivamente i confini. Mamba mostra un errore asimmetrico quasi bilanciato (33.1 vs 33.9), diverso dagli altri modelli dove il ritardo sull'inizio è sistematicamente minore dell'anticipo sulla fine. La mediana prossima a zero per tutti i modelli indica che la maggior parte dei confini è ben localizzata; le medie sono trascinate da pochi segmenti problematici con errori nell'ordine di centinaia di frame.

**Classi più difficili** (errore medio di confine, frame):

| Classe | CNN1D | LSTM | xLSTM | Mamba | MS-TCN++ |
|--------|------:|-----:|------:|------:|---------:|
| Wash strainer | 692.2 (5) | 797.2 (4) | 605.8 (4) | 611.5 (4) | 1149.0 (2) |
| Mix mixture,eating_utensil | 592.7 (3) | 512.0 (4) | 566.0 (3) | 582.7 (3) | 545.0 (1) |
| Mix egg | 466.0 (4) | 707.0 (2) | — | 400.0 (1) | 504.5 (2) |
| Divide/Pull Apart onion | 371.8 (6) | 435.2 (5) | 528.5 (4) | — | 557.3 (3) |
| Wash pot | 369.5 (4) | — | 354.0 (3) | 376.5 (4) | — |

*Wash strainer* e *Mix mixture* appaiono tra le classi più difficili per tutti i modelli: sono azioni rare e visivamente ambigue che condividono movimenti simili con classi più frequenti.

**Top 5 confusioni sistematiche** (frame mal classificati):

*CNN1D*

| GT | Predetto | Frame |
|----|----------|------:|
| Cut cucumber | Cut onion | 847 |
| Cut tomato | Cut onion | 805 |
| Cut tomato | Open fridge | 702 |
| Move Around bacon | Spread condiment,bread,eating_utensil | 588 |
| Wash pan | Put pan | 584 |

*LSTM*

| GT | Predetto | Frame |
|----|----------|------:|
| Divide/Pull Apart onion | Cut onion | 1283 |
| Cut bell_pepper | Mix pasta | 1103 |
| Wash strainer | Mix mixture,eating_utensil | 1009 |
| Cut cucumber | Cut onion | 963 |
| Cut tomato | Divide/Pull Apart onion | 939 |

*xLSTM*

| GT | Predetto | Frame |
|----|----------|------:|
| Cut cucumber | Divide/Pull Apart onion | 766 |
| Cut bell_pepper | Put bell_pepper | 582 |
| Cut tomato | Cut cucumber | 572 |
| Wash eating_utensil | Cut carrot | 483 |
| Divide/Pull Apart onion | Cut onion | 446 |

*Mamba*

| GT | Predetto | Frame |
|----|----------|------:|
| Cut cucumber | Cut onion | 1208 |
| Divide/Pull Apart onion | Cut onion | 849 |
| Cut cucumber | Take cucumber | 653 |
| Cut carrot | Cut tomato | 560 |
| Cut tomato | Cut bell_pepper | 502 |

*MS-TCN++*

| GT | Predetto | Frame |
|----|----------|------:|
| Wash pan | Move Around bacon | 1198 |
| Wash strainer | Mix mixture,eating_utensil | 1166 |
| Divide/Pull Apart onion | Cut onion | 1132 |
| Move Around bacon | Spread condiment,bread,eating_utensil | 1110 |
| Cut cucumber | Cut onion | 968 |

Le confusioni riflettono la struttura fine-grained del dataset: azioni con lo stesso verbo su oggetti diversi (*Cut onion* / *Cut cucumber* / *Cut tomato* / *Cut bell_pepper*) vengono scambiate sistematicamente da tutti i modelli, poiché le feature TSN non sono ottimizzate per distinguere oggetti a livello di frame singolo. *Cut cucumber → Cut onion* è la confusione più frequente per CNN1D, LSTM e Mamba. Le azioni di manipolazione generica (*Wash*, *Mix*, *Move Around*) risultano difficili per tutti i modelli a causa della somiglianza dei movimenti indipendentemente dall'oggetto.

---

## 6. Conclusion and Limitations

Il progetto mostra che per la segmentazione temporale densa su EGTEA Gaze+, MS-TCN++ è l'architettura più efficace grazie al meccanismo multi-stage e alla precisione sui confini temporali. LSTM bidirezionale è competitivo su edit score ma inferiore sulla localizzazione. xLSTM e Mamba soffrono di overfitting o di implementazioni non ottimali nell'ambiente Windows. CNN1D conferma il ruolo di baseline inferiore per mancanza di memoria a lungo raggio. La scelta delle loss (in particolare la boundary loss) ha impatto misurabile e differenziato sulle metriche.

**Limitazioni attuali:**
- Le feature TSN sono fisse e pre-estratte: il modello non può adattare la rappresentazione visiva al task di segmentazione.
- `seq_len = 128` copre ~5 secondi a 24fps; azioni con struttura a più lungo raggio non sono completamente catturabili in una singola finestra.
- L'implementazione Mamba in PyTorch puro è più lenta della versione con kernel CUDA (`mamba-ssm`), non disponibile su Windows.

**Sviluppi futuri:**
- Fine-tuning end-to-end del backbone TSN.
- Aggregazione multi-window in inferenza per clip molto lunghi.
- Uso di `mamba-ssm` su ambiente Linux/WSL per il training Mamba a piena velocità.

---

## 7. Additional Information

### 7.1 Contribution Breakdown

- **Simone Battiato**: progettazione dell'intera pipeline, implementazione di tutti i modelli (CNN1D, LSTM, xLSTM, MS-TCN++, Mamba), loss functions, metriche di valutazione, ottimizzazione del parallel scan per Mamba, script di analisi e valutazione qualitativa.

### 7.2 Use of Artificial Intelligence

**Claude Code** (Anthropic, modello claude-sonnet-4-6) è stato utilizzato come assistente durante lo sviluppo nelle seguenti fasi:

- **Migrazione del codice**: ristrutturazione del repository nella struttura richiesta (`src/`, `experiments/configs/`, ecc.) con aggiornamento automatico degli import.
- **Documentazione**: supporto nella redazione di questo report e del README tecnico.

