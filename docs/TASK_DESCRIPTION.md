# Temporal Action Segmentation — Descrizione del Task

## 1. Definizione del Problema

La segmentazione temporale delle azioni (*temporal action segmentation*) è il problema di assegnare un'etichetta di azione a ogni singolo frame di un video. Formalmente, dato un video rappresentato come una sequenza di vettori di feature **F** = {**f**₁, **f**₂, …, **f**_T} con **f**_t ∈ ℝ^d, l'obiettivo è produrre una sequenza di predizioni Ŷ = {ŷ₁, ŷ₂, …, ŷ_T} dove ogni ŷ_t ∈ {0, 1, …, C} indica l'azione in corso al frame t, con 0 che rappresenta il background e C il numero di classi azione.

A differenza della classificazione di clip (*action recognition*), che produce una sola etichetta per l'intero segmento video, la segmentazione temporale richiede una predizione densa frame per frame, catturando sia il contenuto di ogni istante sia la struttura temporale della sequenza. A differenza della rilevazione di azioni (*action detection*), che individua solo i segmenti di interesse, la segmentazione temporale copre l'intera durata del video, incluse le fasi di background tra un'azione e l'altra.

### Relazione con task simili

| Task | Input | Output | Differenza principale |
|---|---|---|---|
| Action recognition | clip intera | 1 label | nessuna struttura temporale |
| Action detection | video lungo | lista di segmenti con label e timestamp | non copre il background |
| **Action segmentation** | sequenza di frame | 1 label per frame | copertura densa, include background |
| Action anticipation | osservazione parziale | label futura | predice cosa non è ancora avvenuto |

### Sfide

**Sbilanciamento delle classi.** I clip EGTEA contengono una quota rilevante di background prima e dopo l'azione principale. Senza opportuni accorgimenti, un modello tende a predire background per la maggior parte dei frame, ottenendo alta accuracy complessiva senza aver appreso le classi di interesse.

**Granularità fine delle classi.** Le 106 classi sono combinazioni di 19 verbi e 51 oggetti. Azioni come *Cut tomato* e *Cut bell pepper* coinvolgono lo stesso movimento della mano su oggetti diversi, rendendone la distinzione difficile a partire da feature RGB senza informazioni esplicite sull'oggetto manipolato.

**Ambiguità dei confini temporali.** L'inizio e la fine di un'azione non sono sempre netti: la transizione tra due azioni consecutive avviene in modo graduale, e le annotazioni stesse riflettono una valutazione soggettiva del momento esatto di cambio.

**Dipendenze temporali a lungo raggio.** Alcune azioni seguono sequenze ricorrenti (aprire → prendere → tagliare → mescolare). Modellare queste dipendenze richiede un contesto temporale che può estendersi su decine o centinaia di frame.

### Applicazioni

La segmentazione temporale delle azioni trova applicazione in scenari che richiedono la comprensione continua del comportamento umano: sistemi di assistenza per persone anziane o con disabilità, monitoraggio automatico di procedure industriali e chirurgiche, analisi di workflow in ambienti di produzione, e interfacce uomo-robot che devono interpretare le intenzioni dell'operatore in tempo reale.

---

## 2. Dataset: EGTEA Gaze+

EGTEA Gaze+ è un dataset egocentric acquisito con una telecamera montata sulla testa di 32 soggetti durante sessioni di preparazione pasti in cucina. Il dataset comprende 86 sessioni video per un totale di circa 28 ore di contenuto a 24 fps.

Le azioni sono annotate con 106 classi composte da combinazioni verbo-oggetto (es. *Cut tomato*, *Mix egg*, *Pour water*) più una classe background. In totale sono presenti circa 10.325 istanze di azione, distribuite su 7 tipi di pasto.

### Struttura degli split

Il dataset è organizzato in tre fold di cross-validation. In questo lavoro viene utilizzato lo split 1:

| Split | Clip | Modalità |
|---|---:|---|
| Train | 6.277 | random crop per epoch |
| Val | 2.022 | sliding window |
| Test | 2.022 | sliding window |

La validazione è estratta dai clip di training con seed fisso (val_size=2022), a livello di clip, garantendo che nessun frame di training compaia in validazione.

### Feature

Le feature visive sono pre-estratte con un backbone **TSN** (*Temporal Segment Network*) addestrato sul dataset stesso, producendo vettori di dimensione d = 1024 per ogni frame, salvati in un archivio LMDB. Il modello non elabora mai i pixel grezzi del video, ma opera esclusivamente su queste rappresentazioni compresse.

### Modalità di campionamento

- **Training — random crop**: ad ogni epoca viene estratta una finestra casuale di `seq_len=128` frame da ciascun clip. Introduce variabilità sulla posizione della finestra, equivalente a data augmentation temporale.
- **Validation/Test — sliding window**: l'intera durata del clip viene coperta con finestre sovrapposte di `seq_len=128` frame e `stride=64` (overlap 50%), garantendo che ogni frame venga valutato.

```
Clip:  |-----------------------------------------------|
       seq_len=128          stride=64

Win 1: |-----128-----|
Win 2:         |-----128-----|
Win 3:                 |-----128-----|
...
```

---

## 3. Funzioni di Loss

La loss totale combina tre termini:

```
L = L_CE + λ_s · L_smooth + λ_b · L_boundary
```

con `λ_s = 0.2` e `λ_b = 0.3`.

### 3.1 Cross-Entropy con Label Smoothing e Class Weights

La CE standard per classificazione frame per frame su C+1 classi.

**Sbilanciamento.** Il background domina i clip EGTEA. Senza correzione, il modello impara a predire quasi sempre background ottenendo alta accuracy senza apprendere le azioni. Il parametro `bg_weight=0.05` riduce il peso del background nella loss, forzando il modello a concentrarsi sulle classi foreground.

**Label smoothing** (ε=0.1). Distribuisce una piccola probabilità su tutte le classi invece di concentrarla a 1.0 sulla classe corretta. Riduce l'overconfidence del modello e migliora la generalizzazione sulle classi visivamente simili.

### 3.2 Smooth Loss (MS-TCN style)

```
L_smooth = mean( clamp((log_p[t] − log_p[t−1])², max=16) · mask[t] )
```

dove `mask[t] = 1` se il GT non cambia tra il frame t−1 e il frame t.

**Perché serve.** La CE tratta ogni frame in modo indipendente e non impone alcun vincolo di coerenza temporale. Il risultato è una sequenza di predizioni "rumorosa", con transizioni spurie che non esistono nel GT (flickering). La smooth loss penalizza cambiamenti bruschi nelle distribuzioni di probabilità predette nei frame in cui il GT rimane costante.

Opera in spazio log-prob perché in quel dominio le variazioni sono più uniformi tra le classi. Il clamp a 16 evita che pochi frame ad alta varianza dominino il gradiente. Il risultato pratico è che i segmenti predetti diventano più compatti e coerenti, con impatto diretto su edit score e F1@k.

### 3.3 Boundary Loss

```
L_boundary = CE(logits[boundary_mask], targets[boundary_mask])
```

dove `boundary_mask` seleziona i frame entro ±k frame da ogni transizione GT.

**Perché serve.** La CE pesa tutti i frame allo stesso modo, ma un errore al confine di un'azione (frame 47 invece di frame 45) è concettualmente diverso da un errore nel mezzo di un segmento. La boundary loss aumenta il peso della CE vicino ai confini delle azioni, incentivando il modello a essere preciso sul "quando inizia e finisce" ogni azione. Ha impatto diretto su boundary_f1 e F1@50.

---

## 4. Metriche di Valutazione

L'accuracy frame-level da sola non è sufficiente per valutare la segmentazione temporale: se il 60% dei frame è background, un modello che predice sempre background ottiene accuracy=60% senza aver appreso nulla. Servono metriche che misurano la struttura temporale.

### 4.1 mIoU (mean Intersection over Union)

Per ogni classe foreground c calcola:

```
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

a livello di frame, poi fa la media tra tutte le classi foreground. Esclude il background dal calcolo. Punisce sia i falsi positivi (predici un'azione che non c'è) sia i falsi negativi (manchi l'azione).

### 4.2 Edit Score (distanza di Levenshtein)

Converte le sequenze di predizioni e GT in sequenze di segmenti (ignorando il background), poi calcola la distanza di Levenshtein minima tra le due sequenze di etichette. Normalizzata:

```
edit_score = 1 − edit_distance / max(|pred_segments|, |gt_segments|)
```

Misura la correttezza dell'ordine delle azioni, non frame per frame. Cattura errori strutturali che mIoU non vede: un modello che predice le azioni nell'ordine sbagliato viene penalizzato anche se la copertura di frame è alta.

### 4.3 F1@k (con soglia di overlap)

Per ogni coppia (segmento predetto, segmento GT) della stessa classe, calcola l'overlap temporale. Un segmento predetto è TP se il suo overlap con il GT corrispondente supera la soglia k%. Calcola poi precision e recall sui segmenti (non sui frame) e da lì F1.

Disponibile con k = 10, 25, 50. F1@10 è permissivo, F1@50 è severo. Una grande differenza tra F1@10 e F1@50 indica che i segmenti vengono trovati approssimativamente ma localizzati male nel tempo.

### 4.4 Boundary F1

Conta quanti confini predetti (transizioni tra classi) cadono entro una tolleranza di 2 frame da un confine GT. Calcola precision e recall sui confini:

- **Precision**: quanti boundary predetti sono corretti
- **Recall**: quanti boundary GT sono stati trovati

Misura direttamente la qualità della localizzazione temporale dei confini, ed è il complemento diretto della boundary loss.

### 4.5 acc_fg

Accuracy calcolata solo sui frame foreground (esclude il background). Misura quanto bene il modello discrimina tra le 106 azioni quando una è effettivamente in corso, senza il bias dello sbilanciamento.

### Riepilogo

| Metrica | Cosa misura | Sensibile a |
|---|---|---|
| mIoU | qualità assoluta per classe | falsi positivi e negativi |
| edit_score | struttura e ordine delle azioni | sequenza sbagliata |
| F1@10/25/50 | presenza e localizzazione dei segmenti | overlap temporale |
| boundary_f1 | precisione dei confini temporali | offset di inizio/fine |
| acc_fg | discriminazione tra classi foreground | confusione tra classi simili |

---

## 5. Sintesi

La segmentazione temporale delle azioni su EGTEA Gaze+ è un problema di classificazione densa su sequenze temporali con tre difficoltà principali: sbilanciamento verso il background, alta granularità delle classi azione, e necessità di predizioni coerenti nel tempo. Le tre loss affrontano rispettivamente la classificazione corretta (CE), la coerenza temporale (smooth), e la precisione dei confini (boundary). Le cinque metriche misurano aspetti complementari della qualità della segmentazione, dalla copertura dei frame alla struttura della sequenza.
