# process-constraint-steering

Strumenti per studiare il **task-vector steering nella predizione dei suffissi di processo**, orientando un modello verso il rispetto di vincoli dichiarativi DECLARE.

Il progetto si basa su **LUPIN** di Pasquadibisceglie, Appice e Malerba. LUPIN è il modello di riferimento; questo repository raccoglie le funzionalità di adattamento ai vincoli, steering, valutazione e interpretazione delle direzioni.

Dato un checkpoint base `θ₀` e un checkpoint adattato `θ₁`, il task vector è `τ = θ₁ − θ₀`. Lo steering costruisce `θ(α) = θ₀ + ατ`: una volta ottenuti i due checkpoint, cambiare `α` non richiede un nuovo fine-tuning.

## Indice

- [Preparazione dell'ambiente](#preparazione-dellambiente)
- [Dati e struttura del progetto](#dati-e-struttura-del-progetto)
- [Panoramica dei comandi](#panoramica-dei-comandi)
- [Estrazione delle process execution da OCEL](#estrazione-delle-process-execution-da-ocel)
- [1. Addestramento del modello base](#1-addestramento-del-modello-base)
- [2. Fine-tuning su tracce conformi](#2-fine-tuning-su-tracce-conformi)
- [3. Fine-tuning con loss semantica](#3-fine-tuning-con-loss-semantica)
- [4. Valutazione di un checkpoint](#4-valutazione-di-un-checkpoint)
- [5. Steering lungo una direzione](#5-steering-lungo-una-direzione)
- [6. Composizione di due direzioni](#6-composizione-di-due-direzioni)
- [7. Interpretazione delle direzioni](#7-interpretazione-delle-direzioni)
- [Metriche](#metriche)
- [Estrazione dei vincoli](#estrazione-dei-vincoli)
- [Riferimenti](#riferimenti)

## Preparazione dell'ambiente

Eseguire tutti i comandi **dalla radice del repository**. Gli input e gli output relativi vengono risolti rispetto alla directory corrente.

```bash
git clone https://github.com/godSaveDaniele/process-constraint-steering.git
cd process-constraint-steering
python3.10 -m venv .venv
source .venv/bin/activate
```

Se l'ambiente `.venv` esiste già, basta attivarlo. Il codice usa CUDA quando disponibile e CPU altrimenti. Modello e tokenizer `prajjwal1/bert-medium` vengono caricati tramite Hugging Face: al primo utilizzo occorre scaricarli, oppure averli già nella cache.

## Dati e struttura del progetto

```text
cli/                    Comandi di training, valutazione, steering e interpretazione
constraints/            Valutatori DECLARE ed estrazione dei vincoli dai log
semantic_loss/          Loss differenziabili per Response e Chain Response
neural_network/         Modello multi-output e dataset per PyTorch
preprocessing/          Conversione dei log in storie testuali e coppie prefisso-suffisso
utility/                Configurazioni dei dataset e gestione del seed
event_log/              CSV di input
log_history/            Dati preprocessati e mapping delle attività
models/                 Checkpoint locali
outputs/                Risultati di valutazione e analisi
```

I CSV disponibili nella copia corrente sono `event_log/bpic2017_o.csv` e `event_log/logistics.csv`.  

Il preprocessing richiede colonne come `case`, `activity`, `timestamp` e gli attributi previsti dalla configurazione del dataset. Un nuovo CSV non è automaticamente compatibile: occorre definirne attributi e template narrativi in `utility/log_config.py`. Il preprocessing narrativo legge Logistics come CSV di tracce lineari. Il comando `cli.extract_process_execution_OCEL` genera questo CSV a partire dall’OCEL 2.0 originale.

`Log(dataset, "all")` in `preprocessing/log_to_history.py`:

- normalizza le etichette delle attività rimuovendo spazi, `+`, `-` e `_`;
- ordina le istanze per timestamp iniziale e assegna il primo 66% al training e il restante 34% al test;
- genera le coppie prefisso-suffisso e le descrizioni testuali;
- salva i file `<dataset>_<tipo>_all.pkl` in `log_history/<dataset>/`, inclusi mapping, suffissi, lunghezze e attività dei prefissi.

I comandi di training eseguono il preprocessing. La valutazione e lo steering richiedono i file già generati, inclusi `prefix_activities_test`: i vecchi dati preprocessati potrebbero non contenerli. Per generarli senza addestrare il modello:

```bash
python -c 'from preprocessing.log_to_history import Log; Log("bpic2017_o", "all")'
```



I checkpoint devono avere architettura, mapping delle attività e numero di teste compatibili con i dati preprocessati. `models/` e i file `.pth` sono esclusi dal versionamento: dopo il clone occorre addestrare il modello o fornire checkpoint compatibili.

## Panoramica dei comandi

| Modulo | Funzionalità |
| --- | --- |
| `cli.extract_process_execution_OCEL` | Conversione del Logistics OCEL 2.0 in tracce CSV |
| `cli.main` | Addestramento del modello base |
| `cli.constrained_fine_tuning` | Fine-tuning su tracce conformi o su un sottoinsieme casuale di controllo |
| `cli.semantic_fine_tuning` | Fine-tuning con sola loss semantica oppure CE + loss semantica |
| `cli.eval_model` | Valutazione predittiva e di conformità di un checkpoint |
| `cli.weight_steering` | Valutazione lungo una direzione al variare di α |
| `cli.combined_steering` | Valutazione di una griglia di combinazioni di due direzioni |
| `cli.interpret_directions` | Analisi dei logit per una direzione heads-only e un vincolo Response |



## Estrazione delle process execution da OCEL

`cli.extract_process_execution_OCEL`  estrae una process execution per ogni oggetto Customer Order
e produce il CSV utilizzato dal preprocessing Logistics. Non è un estrattore
generico per qualsiasi OCEL: tipi di oggetto e qualificatori sono quelli del
dataset **Container Logistics**.


```bash
python -m pip install pm4py
python -m cli.extract_process_execution_OCEL --help
```

Fornire il percorso del dataset originale, che non viene copiato nel repository:

```bash
python -m cli.extract_process_execution_OCEL \
  --input /percorso/container_logistics.json \
  --output event_log/logistics.csv \
  --report-dir outputs/extract_ocel_logistics
```

`--input` è obbligatorio e richiede un JSON OCEL 2.0. `--output` vale
`event_log/logistics.csv` per default; `--report-dir` vale
`outputs/extract_ocel_<nome CSV>`. Le directory vengono create quando necessario;
un CSV o un report già presenti nella destinazione vengono sovrascritti.

### Regole di estrazione

A partire dal Customer Order, le relazioni oggetto-oggetto vengono percorse
nell'ordine seguente, verificando qualificatore e tipo dell'oggetto raggiunto:

| Oggetto di partenza | Oggetto raggiunto | Qualificatore | Direzione della relazione OCEL |
| --- | --- | --- | --- |
| Customer Order | Transport Document | `TD for CO` | Diretta |
| Transport Document | Container | `CR for TD` | Inversa |
| Container | Handling Unit | `CR contains HU` | Diretta |
| Container | Truck | `TR loads CR` | Inversa |
| Transport Document | Vehicle | `Ersatz VH for TD` | Diretta |
| Transport Document | Vehicle | `Regular VH for TD` | Diretta |
| Transport Document | Vehicle | `High-Prio VH for TD` | Diretta |

Gli eventi selezionati sono quelli collegati a Customer Order, Transport Document,
Container e Handling Unit raggiunti. Truck e Vehicle non ampliano direttamente
l'insieme degli eventi. Ogni evento compare una sola volta nella stessa execution;
gli eventi vengono ordinati per timestamp e identificativo. Non vengono applicati
filtri temporali sui cicli di vita degli oggetti.

### CSV e diagnostiche

Il CSV contiene `case`, `activity`, `timestamp`, `resource`, `event_id` ed
`event_objects`. `case` è l'ID del Customer Order; `resource` vale `UNK`.
`event_objects` contiene un array JSON con ID, tipo e qualificatore di **tutte**
le relazioni dell'evento, anche con oggetti esterni all'execution. Queste informazioni
alimentano i template narrativi già presenti in `preprocessing/logistics_narrative.py`.

La directory delle diagnostiche contiene:

- `report.json`: dimensioni, lunghezze delle tracce, eventi condivisi e sovrapposizione
  rispetto allo split cronologico 66%/34% usato dal preprocessing;
- `trace_lengths.csv`: numero di eventi di ogni execution;
- `shared_events_train_test.csv`: righe delle execution che contengono eventi
  presenti sia nella partizione di training sia in quella di test.

Il flattening preserva gli eventi condivisi: il report ne misura la presenza,
ma non modifica lo split e non elimina automaticamente la sovrapposizione.

Dopo l'estrazione si può generare il preprocessing senza addestrare il modello:

```bash
python -c 'from preprocessing.log_to_history import Log; Log("logistics", "all")'
```

Oppure avviare il training, che esegue anche il preprocessing:

```bash
python -m cli.main logistics --seed 42
```

Per uso programmatico, il nuovo modulo espone `export_logistics(input_path,
output_path, report_dir=None)`, che restituisce il DataFrame e il report, e
`extract_logistics_process_execution(ocel, customer_order_id)`, che restituisce
oggetti per tipo ed eventi della singola execution.

## 1. Addestramento del modello base

```bash
python -m cli.main bpic2017_o --seed 42
```

`dataset` è posizionale; `--seed` è opzionale e vale `42` per default. Il comando rigenera il preprocessing e addestra BERT con le teste di classificazione per le posizioni future.

Le impostazioni attuali sono definite nel codice: massimo 512 token, batch size 16, learning rate `1e-5`, massimo 50 epoche ed early stopping dopo 5 epoche senza miglioramento della validation loss. Non sono opzioni della CLI.

**Output:** `models/bpic2017_o/seed_42/`, con `epoch_001.pth`, ecc. e `best.pth`. Il comando richiede che la directory di output non esista già.

Nei comandi successivi passare esplicitamente `--checkpoint models/bpic2017_o/seed_42/best.pth`: il percorso di fallback di alcuni comandi è ancora `models/<dataset>_all.pth`, diverso da quello prodotto dal training corrente.

## 2. Fine-tuning su tracce compliant

Seleziona le tracce di training che soddisfano il vincolo, incluse quelle vacue, e ottimizza la cross-entropy sui relativi suffissi.

```bash
python -m cli.constrained_fine_tuning \
  --dataset bpic2017_o \
  --checkpoint models/bpic2017_o/seed_42/best.pth \
  --template Response --activation Returned --target Accepted \
  --mode heads_only --epochs 3 --batch-size 16 --seed 42
```

`--dataset`, `--template`, `--activation` e `--target` sono obbligatori. I template ammessi sono `Response`, `Precedence` e `"Chain Response"`.

Per il controllo su un sottoinsieme casuale, aggiungere `--random-subset` allo stesso comando. Viene estratto dal training originale lo stesso numero di **casi** del sottoinsieme conforme, senza richiedere che soddisfino il vincolo; il numero di coppie prefisso-suffisso può differire.

### Opzioni comuni ai due comandi di fine-tuning

| Opzione | Default | Significato |
| --- | --- | --- |
| `--checkpoint` | `models/<dataset>_all.pth` | Checkpoint da adattare |
| `--mode` | `heads_only` | Parametri aggiornabili |
| `--top-k` | non impostato | Numero di blocchi superiori, obbligatorio solo con `--mode top_k` |
| `--learning-rate` | `1e-5` | Learning rate AdamW |
| `--epochs` | `3` | Numero di epoche |
| `--batch-size` | `8` | Dimensione del batch |
| `--seed` | `42` | Seed per split e operazioni casuali |

Le modalità sono:

- `heads_only`: aggiorna soltanto le teste di classificazione;
- `top_k`: aggiorna le teste, gli ultimi K blocchi Transformer e il pooler;
- `full_short`: aggiorna tutti i parametri. Il numero di epoche resta quello di `--epochs`.

**Output:** `models/<dataset>/constrained/<mode>_seed<seed>/`, oppure `models/<dataset>/random_subset/<mode>_seed<seed>/`. Contiene checkpoint di ogni epoca, `best.pth`, `config.json`, `history.json` e `id2label.pkl`. Il miglior checkpoint è scelto sulla validation loss. I dati selezionati sono salvati nelle corrispondenti sottocartelle di `log_history/<dataset>/`.

La directory di output non include il vincolo: eseguire adattamenti diversi con lo stesso dataset, modalità e seed può sovrascrivere i risultati. Conservare una copia della directory prima di riutilizzare la stessa configurazione.

## 3. Fine-tuning con loss semantica

Utilizza il training originale, senza filtrarlo in base al vincolo. Sono implementate loss semantiche per `Response` e `"Chain Response"`; `Precedence` non è supportato da questo comando.

```bash
python -m cli.semantic_fine_tuning \
  --dataset bpic2017_o \
  --checkpoint models/bpic2017_o/seed_42/best.pth \
  --template Response --activation Returned --target Accepted \
  --mode heads_only --epochs 3 --batch-size 16 --seed 42
```

Per default ottimizza **soltanto la loss semantica**. La cross-entropy viene comunque monitorata. Per ottimizzare `CE + λ × loss_semantica`, aggiungere:

```text
--dual-objective --lambda 1.0
```

`--lambda` vale `1.0` per default e ha effetto sull'obiettivo solo con `--dual-objective`. `--template` è opzionale e vale `Response`; dataset, activation e target sono obbligatori. Le altre opzioni coincidono con quelle della tabella del fine-tuning.

**Output:** `models/<dataset>/semantic_<template>/<mode>_seed<seed>/`, dove i template diventano `response` o `chain_response`. Contiene checkpoint delle epoche, `best.pth`, configurazione, mapping e andamento delle loss totale, CE e semantica. Il checkpoint migliore minimizza la validation loss totale dell'obiettivo scelto.

La directory non distingue coppia di attività, obiettivo puro/combinato o valore di λ. Per confrontare queste varianti, conservare ciascuna directory prima di avviare la successiva. Non passare lo stesso percorso come due checkpoint diversi dopo averlo sovrascritto.

## 4. Valutazione di un checkpoint

```bash
python -m cli.eval_model bpic2017_o \
  --checkpoint models/bpic2017_o/seed_42/best.pth \
  --template Response --activation Returned --target Accepted
```

`dataset` è posizionale. Template, activation e target sono obbligatori; `--checkpoint` usa il fallback descritto sopra. `--limit N` limita l'analisi ai primi N esempi del test; senza limite viene valutato l'intero test.

La decodifica sceglie l'argmax di ciascuna testa e tronca il suffisso alla prima `ENDactivity`. Il vincolo viene valutato su **prefisso + suffisso**, sia per la predizione sia per la traccia reale, senza END o padding.

**Output:** `outputs/eval_<dataset>_<timestamp>/`:

- `examples.jsonl`: un record per esempio con indice, prefisso, suffisso reale e predetto, stati del vincolo, DLS ed exact match;
- `metrics.json`: metriche aggregate delle predizioni e delle tracce reali, DLS globale e sul sottoinsieme reale conforme.

Le metriche aggregate vengono anche stampate nel terminale. I template disponibili sono `Response`, `Precedence` e `"Chain Response"`.

## 5. Steering lungo una direzione

```bash
python -m cli.weight_steering \
  --dataset bpic2017_o \
  --base-checkpoint models/bpic2017_o/seed_42/best.pth \
  --constrained-checkpoint models/bpic2017_o/constrained/heads_only_seed42/best.pth \
  --template Response --activation Returned --target Accepted \
  --alphas 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
```

Tutti i parametri dell'esempio tranne `--alphas` sono obbligatori. Il checkpoint adattato può provenire da una qualsiasi delle modalità di fine-tuning.

- `α = 0`: modello base;
- `α = 1`: modello adattato;
- `0 < α < 1`: interpolazione;
- `α > 1`: estrapolazione oltre il checkpoint adattato.

La griglia predefinita è `0 0.1 0.2 0.3 0.75 1`. `--limit N` limita gli esempi valutati per ogni α. Sono supportati gli stessi tre template della valutazione singola.

**Output:** `outputs/steering_<dataset>_<timestamp>/`, con `config.json`, `results.csv` e `metrics.json`. Ogni α genera inoltre una directory `eval_*` con i dettagli per esempio. Il checkpoint interpolato è temporaneo: non viene mantenuta una copia `.pth` per ciascun α.

Le metriche vengono calcolate direttamente durante l'inferenza; non serve più uno script separato di rivalutazione.

## 6. Composizione di due direzioni

Combina due adattamenti dello **stesso checkpoint base**:

```text
θ(α, β) = θ₀ + α(θ₁ − θ₀) + β(θ₂ − θ₀)
```

Prima di eseguire l'esempio, preparare due checkpoint distinti e conservarli nei percorsi scelti. Nell'esempio `checkpoints/c1.pth` e `checkpoints/c2.pth` sono percorsi illustrativi, non file forniti dal repository.

```bash
python -m cli.combined_steering \
  --dataset bpic2017_o \
  --base-checkpoint models/bpic2017_o/seed_42/best.pth \
  --checkpoint-1 checkpoints/c1.pth \
  --checkpoint-2 checkpoints/c2.pth \
  --template-1 Response --activation-1 Returned --target-1 Accepted \
  --template-2 Response --activation-2 "Sent (mail and online)" --target-2 Returned \
  --alphas 0 0.5 1 --betas 0 0.5 1 --batch-size 16
```

Dataset, tre checkpoint e le coppie activation/target sono obbligatori. `--template-1` e `--template-2` valgono `Response` per default e accettano anche `Precedence` e `"Chain Response"`. Le griglie predefinite sono entrambe `0 0.1 0.2 0.3`, il batch size è `16`; `--limit N` limita il test.

**Output:** `outputs/combined_steering_<dataset>_<timestamp>/` con configurazione, CSV e JSON aggregati. Ogni coppia `(α, β)` ha una sottocartella `pair_000`, ecc., con `metrics.json` ed `examples.jsonl`.

La valutazione riporta DLS globale, metriche di ciascun vincolo, compliance congiunta e conteggi `both`, `only_1`, `only_2`, `neither`. La conformità congiunta include il soddisfacimento vacuo. Questo comando non calcola attualmente DLS sul sottoinsieme conforme o numero medio di attivazioni.

## 7. Interpretazione delle direzioni

Analizza un adattamento **heads-only**, distinguendo lo stato di `Response(A, B)` nel prefisso:

- `satisfied`: A è presente e le sue obbligazioni sono già soddisfatte;
- `pending`: rimane un'obbligazione pendente;
- `absent`: A non è presente.

```bash
python -m cli.interpret_directions \
  --dataset bpic2017_o \
  --base-checkpoint models/bpic2017_o/seed_42/best.pth \
  --destination-checkpoint models/bpic2017_o/constrained/heads_only_seed42/best.pth \
  --activation Returned --target Accepted --batch-size 16
```

Dataset, checkpoint e attività sono obbligatori. Il batch size predefinito è `16`. `--regenerate-logs` forza la rigenerazione del preprocessing, effettuata anche se mancano i file richiesti.

Il comando verifica che i parametri esterni alle teste siano invariati e calcola `Δz = ΔW h + Δb` e i logit base `z₀ = W₀ h + b₀`, senza softmax. Riporta medie per gruppo su **tutte le posizioni future e le attività, END incluso**, dando lo stesso peso a ogni prefisso. La variazione corrisponde al passaggio al checkpoint di destinazione (`α = 1`); non c'è un parametro α nella CLI.

**Output:** `outputs/interpret_direction_<dataset>_<timestamp>/`:

- `config.json` e `prefix_groups.csv`;
- `<gruppo>_mean_base_logits.csv` e `<gruppo>_mean_delta_logits.csv`;
- `mean_base_logits.png/.pdf` e `mean_delta_logits.png/.pdf`.

Si tratta di matrici e mappe di calore dei logit, non di metriche di probabilità. L'analisi usa la semantica di Response e non espone un'opzione per altri template.

## Metriche

Le unità della valutazione sono le **coppie prefisso-suffisso**: prefissi diversi della stessa istanza contribuiscono come esempi distinti. Indicando con F, V e U il numero di esempi fulfilled, violated e vacuous, `N = F + V + U`:

| Metrica | Definizione |
| --- | --- |
| `compliance` | `(F + U) / N` |
| `non_vacuous_compliance` | Supporto non vacuo: `F / N` |
| `confidence` | `F / (F + V)`; `null` se non vi sono attivazioni |
| `activation_rate` | `(F + V) / N` |
| `fulfilled`, `violated`, `vacuous` | Conteggi F, V e U |
| `dl_score_original` | DLS media su tutti gli esempi |
| `dl_score_compliant_subset` | DLS media sugli esempi con traccia **reale** fulfilled o vacuous |
| `compliant_examples` | Numero degli esempi reali conformi, riportato nel JSON della valutazione singola e dello steering |
| `mean_activations_per_trace` | Solo Logistics: media delle occorrenze dell'attività attivante in prefisso + suffisso |

La DLS è `1 − distanza_Damerau_Levenshtein / max(lunghezza_predetta, lunghezza_reale)`, codificando ciascuna attività come un singolo carattere. Se entrambi i suffissi sono vuoti vale `1`. Il sottoinsieme conforme dipende dalla traccia reale e rimane quindi fisso al variare di α; se è vuoto, la relativa DLS è `null`.

Per la media delle attivazioni si conta B in `Precedence(A, B)` e A in Response e Chain Response. Anche questa media è calcolata sulle coppie prefisso-suffisso valutate, non sulle sole istanze uniche del log.

Nel CSV di `weight_steering` il supporto è chiamato `non_vacuous_compliance (support)`, in continuità con il precedente script di rivalutazione. Nei JSON e nelle colonne `c1_*`/`c2_*` della composizione si chiama `non_vacuous_compliance`. I valori non definiti sono `null` nei JSON e celle vuote nei CSV. Le proporzioni sono espresse fra 0 e 1, non in percentuale.

**Risultati storici:** prima dell'aggiornamento delle metriche, `non_vacuous_compliance` indicava invece `F / (F + V)`, oggi denominato `confidence`. I vecchi file non vengono modificati retroattivamente.

## Estrazione dei vincoli

È disponibile anche un'utilità per valutare coppie di attività sulle tracce del log e selezionare i vincoli secondo soglie di confidence, attivazione e violazioni:

```bash
python -m constraints.extraction
```

Questo script non espone opzioni CLI: dataset, template e soglie sono configurati nella funzione `main()` di `constraints/extraction.py`. Attualmente analizza `logistics` con `Chain Response`.

Produce `constraints/outputs/<dataset>_all.csv` e `<dataset>_selected.csv`. Diversamente dalla valutazione dei suffissi, qui le metriche sono calcolate sulle tracce complete raggruppate per `case` nel CSV originale.

## Riferimenti

V. Pasquadibisceglie, A. Appice e D. Malerba, *LUPIN: A LLM Approach for Activity Suffix Prediction in Business Process Event Logs*, ICPM 2024.

- [Paper LUPIN](https://doi.org/10.1109/ICPM63005.2024.10680620)
- [Implementazione originale LUPIN](https://github.com/vinspdb/LUPIN)
