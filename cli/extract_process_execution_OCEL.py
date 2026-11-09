"""Estrae le process execution dal Logistics OCEL 2.0 e le esporta per LUPIN.

Migrazione del flusso di PM4Py_OCEL/notebooks: logistics_process_execution.py
ed export_logistics_lupin.py. Le regole sono specifiche per Container Logistics.
Gli eventi condivisi sono preservati, senza filtri temporali sui cicli degli oggetti.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Hashable, Literal

import pandas as pd

Direction = Literal['forward', 'backward']


@dataclass(frozen=True)
class ObjectRelationRule:
    """Passo di espansione consentito lungo una relazione oggetto-oggetto."""

    source_type: str
    target_type: str
    qualifier: str
    direction: Direction


# L'ordine è significativo: gli oggetti raggiunti alimentano le regole successive.
LOGISTICS_RELATION_RULES: tuple[ObjectRelationRule, ...] = (
    ObjectRelationRule('Customer Order', 'Transport Document', 'TD for CO', 'forward'),
    ObjectRelationRule('Transport Document', 'Container', 'CR for TD', 'backward'),
    ObjectRelationRule('Container', 'Handling Unit', 'CR contains HU', 'forward'),
    ObjectRelationRule('Container', 'Truck', 'TR loads CR', 'backward'),
    ObjectRelationRule('Transport Document', 'Vehicle', 'Ersatz VH for TD', 'forward'),
    ObjectRelationRule('Transport Document', 'Vehicle', 'Regular VH for TD', 'forward'),
    ObjectRelationRule('Transport Document', 'Vehicle', 'High-Prio VH for TD', 'forward'),
)

# Truck e Vehicle sono nel contesto, ma non ampliano direttamente gli eventi.
EVENT_OBJECT_TYPES = frozenset({
    'Customer Order', 'Transport Document', 'Container', 'Handling Unit',
})


@dataclass(frozen=True)
class LogisticsProcessExecution:
    """Oggetti ed eventi della process execution di un Customer Order."""

    customer_order_id: Hashable
    objects_by_type: dict[str, frozenset[Hashable]]
    events: pd.DataFrame


def extract_logistics_process_execution(
    ocel: Any,
    customer_order_id: Hashable,
    objects_by_event: dict | None = None,
) -> LogisticsProcessExecution:
    """Applica le regole Logistics senza filtri temporali sui cicli degli oggetti."""
    object_type_by_id = _object_types(ocel.objects, 'ocel:oid', 'ocel:type')
    _validate_customer_order(object_type_by_id, customer_order_id)
    objects_by_type: dict[str, set[Hashable]] = {
        'Customer Order': {customer_order_id},
    }
    for rule in LOGISTICS_RELATION_RULES:
        source_ids = objects_by_type.get(rule.source_type, set())
        if not source_ids:
            continue
        related_ids = _follow_object_relation(
            object_object_relations=ocel.o2o,
            source_ids=source_ids,
            rule=rule,
            source_column='ocel:oid',
            target_column='ocel:oid_2',
            qualifier_column='ocel:qualifier',
            object_type_by_id=object_type_by_id,
        )
        objects_by_type.setdefault(rule.target_type, set()).update(related_ids)

    selected_object_ids = set().union(*(
        ids for object_type, ids in objects_by_type.items()
        if object_type in EVENT_OBJECT_TYPES
    ))
    event_ids = set(ocel.relations.loc[
        ocel.relations['ocel:oid'].isin(selected_object_ids), 'ocel:eid',
    ].dropna())
    events = ocel.events.loc[ocel.events['ocel:eid'].isin(event_ids)].copy()
    events = events.drop_duplicates(subset=['ocel:eid'])
    events = events.sort_values(
        ['ocel:timestamp', 'ocel:eid'], kind='mergesort', na_position='last',
    ).reset_index(drop=True)
    if objects_by_event is None:
        objects_by_event = build_objects_by_event(ocel)
    events['object_associated'] = [
        list(objects_by_event.get(event_id, ())) for event_id in events['ocel:eid']
    ]
    return LogisticsProcessExecution(
        customer_order_id=customer_order_id,
        objects_by_type={key: frozenset(ids) for key, ids in objects_by_type.items()},
        events=events,
    )


def _follow_object_relation(
    *,
    object_object_relations: pd.DataFrame,
    source_ids: set[Hashable],
    rule: ObjectRelationRule,
    source_column: str,
    target_column: str,
    qualifier_column: str,
    object_type_by_id: dict[Hashable, str],
) -> set[Hashable]:
    if rule.direction == 'forward':
        from_column, to_column = source_column, target_column
    else:
        from_column, to_column = target_column, source_column
    matching = object_object_relations.loc[
        object_object_relations[from_column].isin(source_ids)
        & object_object_relations[qualifier_column].eq(rule.qualifier),
        to_column,
    ].dropna()
    return {
        object_id for object_id in matching.unique()
        if object_type_by_id.get(object_id) == rule.target_type
    }


def _object_types(objects, object_id_column, object_type_column):
    rows = objects[[object_id_column, object_type_column]].dropna().drop_duplicates()
    return dict(rows.itertuples(index=False, name=None))


def _validate_customer_order(object_type_by_id, customer_order_id):
    object_type = object_type_by_id.get(customer_order_id)
    if object_type is None:
        raise ValueError(f'Unknown viewpoint object: {customer_order_id!r}')
    if object_type != 'Customer Order':
        raise ValueError(
            f'Viewpoint object {customer_order_id!r} has type {object_type!r}; '
            "expected 'Customer Order'"
        )


def build_objects_by_event(ocel):
    """Include tutti gli oggetti associati, anche esterni all'execution."""
    return (
        ocel.relations[['ocel:eid', 'ocel:oid']]
        .dropna().drop_duplicates()
        .groupby('ocel:eid', sort=False)['ocel:oid'].agg(list).to_dict()
    )


def export_logistics(input_path, output_path, report_dir=None):
    """Esporta una traccia per Customer Order e le diagnostiche di condivisione."""
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input OCEL e output CSV devono essere file diversi.")
    try:
        import pm4py
    except ModuleNotFoundError as error:
        if error.name != "pm4py":
            raise
        raise ImportError("Installare PM4Py: python -m pip install pm4py") from error


    #Caricamento dell'ocel
    ocel = pm4py.read_ocel2_json(str(input_path))

    # Estrazione object types
    object_types = ocel.objects.set_index('ocel:oid')['ocel:type'].to_dict()

    # Estrazione id degli ordini
    orders = sorted(key for key, value in object_types.items() if value == 'Customer Order')

    # Generazione dizionario evento --> oggetti associati
    objects_by_event = build_objects_by_event(ocel)

    # Relazioni evento-oggetto con qualificatore.
    relations = ocel.relations[['ocel:eid', 'ocel:oid', 'ocel:qualifier']].drop_duplicates()

    # Indicizza le relazioni per evento, preservandone i qualificatori.
    relations_by_event = {}
    for event_id, object_id, qualifier in relations.itertuples(index=False, name=None):
        relations_by_event.setdefault(event_id, []).append({
            'id': object_id,
            'type': object_types[object_id],
            'qualifier': '' if pd.isna(qualifier) else qualifier,
        })

    frames = []
    for index, order in enumerate(orders, 1):
        execution = extract_logistics_process_execution(ocel, order, objects_by_event)

        # Un DataFrame per execution.
        frame = execution.events.rename(columns={
            'ocel:eid': 'event_id', 'ocel:activity': 'activity', 'ocel:timestamp': 'timestamp',
        }).copy()
        if frame.empty:
            raise ValueError(f'Execution senza eventi: {order}')

        #Viene assegnato a tutte le righe del frame l'ordine
        frame['case'] = order
        frame['resource'] = 'UNK'  # Compatibilità con il loader LUPIN.
        frame['event_objects'] = [
            json.dumps(relations_by_event.get(event_id, []), ensure_ascii=False)
            for event_id in frame['event_id']
        ]
        frames.append(frame[['case', 'activity', 'timestamp', 'resource',
                             'event_id', 'event_objects']])
        if index % 100 == 0 or index == len(orders):
            print(f'Execution: {index}/{len(orders)}', flush=True)
    if not frames:
        raise ValueError('Nessun Customer Order nel dataset.')
    data = pd.concat(frames, ignore_index=True)
    if data[['case', 'activity', 'timestamp', 'event_id']].isna().any().any():
        raise ValueError('Identificativi, attività o timestamp mancanti.')
    if data.duplicated(['case', 'event_id']).any():
        raise ValueError('Evento duplicato nella stessa execution.')
    data = data.sort_values(['case', 'timestamp', 'event_id'], kind='mergesort')

    # Stessa selezione cronologica dei casi effettuata da LUPIN.
    starts = data.groupby('case')['timestamp'].min().reset_index()
    starts = starts.sort_values('timestamp', kind='mergesort')
    train_cases = set(starts['case'].iloc[:int(0.66 * len(starts))])
    train = data[data['case'].isin(train_cases)]
    test = data[~data['case'].isin(train_cases)]
    shared = set(train['event_id']) & set(test['event_id'])
    lengths = data.groupby('case').size()
    usage = data.groupby('event_id')['case'].nunique()
    report = {
        'input': str(input_path), 'output': str(output_path),
        'cases': len(starts), 'rows': len(data), 'activities': int(data['activity'].nunique()),
        'original_events': len(ocel.events), 'included_unique_events': int(data['event_id'].nunique()),
        'trace_length_min': int(lengths.min()), 'trace_length_mean': float(lengths.mean()),
        'trace_length_max': int(lengths.max()),
        'events_in_multiple_cases': int((usage > 1).sum()),
        'train_cases': len(train_cases), 'test_cases': len(starts) - len(train_cases),
        'shared_events_train_test': len(shared),
        'test_rows_with_shared_event': int(test['event_id'].isin(shared).sum()),
        'test_cases_with_shared_event': int(test.loc[test['event_id'].isin(shared), 'case'].nunique()),
        'note': 'Flattening preserves event sharing. No temporal filtering of object lifecycles.',
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    report_dir = Path(report_dir) if report_dir is not None else Path('outputs') / f'extract_ocel_{output_path.stem}'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    data[data['event_id'].isin(shared)][['case', 'event_id', 'activity', 'timestamp']].to_csv(
        report_dir / 'shared_events_train_test.csv', index=False)
    lengths.rename('events').to_csv(report_dir / 'trace_lengths.csv')
    print(json.dumps(report, indent=2))
    return data, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='Dataset Container Logistics in formato OCEL 2.0 JSON.')
    parser.add_argument('--output', type=Path, default=Path('event_log/logistics.csv'),
                        help='CSV destinazione (default: event_log/logistics.csv).')
    parser.add_argument('--report-dir', type=Path,
                        help='Directory diagnostiche (default: outputs/extract_ocel_<nome CSV>).')
    args = parser.parse_args()
    export_logistics(args.input, args.output, args.report_dir)


if __name__ == '__main__':
    main()
