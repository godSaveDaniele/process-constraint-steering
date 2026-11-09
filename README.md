# process-constraint-steering

Studio del task-vector steering per la predizione dei suffissi di processo
con vincoli dichiarativi. Il progetto si basa sul modello **LUPIN** di
Pasquadibisceglie, Appice e Malerba e ne studia l’adattamento ai vincoli,
la regolazione mediante task vector e la composizione delle direzioni.

LUPIN rimane il nome del modello di riferimento; il nome di questo progetto
è `process-constraint-steering`.

## Esecuzione locale

Eseguire gli script dalla cartella principale del progetto: i percorsi relativi
per dati, checkpoint e risultati sono risolti rispetto alla directory corrente.

```bash
cd /Users/daniele/Desktop/projects/process-constraint-steering
source .venv/bin/activate
python weight_steering.py --help
```

Per creare un nuovo ambiente:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

La rinomina locale non modifica gli indirizzi dei repository Git remoti.

## Riferimento

V. Pasquadibisceglie, A. Appice e D. Malerba,
“LUPIN: A LLM Approach for Activity Suffix Prediction in Business Process Event Logs”,
ICPM 2024. DOI: 10.1109/ICPM63005.2024.10680620.

Implementazione di riferimento: https://github.com/vinspdb/LUPIN
