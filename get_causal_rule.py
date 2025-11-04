import os
import pickle
from typing import Dict, Optional, Set, Tuple, List

import pandas as pd
import pm4py
import torch
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog, Trace


def load_and_convert_csv_for_all_data(file_path):
    df = pd.read_csv(file_path)
    required_columns = ["case", "activity", "timestamp","resource"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file is missing required columns: {missing_cols}")
    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    column_mapping = {
        "case": "case:concept:name",
        "activity": "concept:name",
        "timestamp": "time:timestamp",
        "resource": "resource:resource_name"
    }
    renamed_columns = {}
    for col in df.columns:
        if col in column_mapping:
            renamed_columns[col] = column_mapping[col]
        else:
            if col.startswith("case:") or col.startswith("time:") or col.startswith("concept:"):
                renamed_columns[col] = col
            else:
                renamed_columns[col] = f"attr:{col}"

    df = df.rename(columns=renamed_columns)
    log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)
    return log


def merge_two_logs(log1: EventLog, log2: EventLog) -> EventLog:
    all_traces = []
    for trace in log1:
        new_trace = Trace(trace, attributes=trace.attributes.copy())
        all_traces.append(new_trace)

    for trace in log2:
        new_trace = Trace(trace, attributes=trace.attributes.copy())
        all_traces.append(new_trace)
    merged_log = EventLog(
        all_traces,
        attributes=log1.attributes if log1.attributes else None,
        extensions=log1.extensions if log1.extensions else None,
        omni_present=log1.omni_present if log1.omni_present else None,
        classifiers=log1.classifiers if log1.classifiers else None,
        properties=log1.properties if hasattr(log1, 'properties') else None
    )

    return merged_log

def discover_and_index_causal_rulesV2(
        log: EventLog,
        activity_to_index: Dict[str, int],
        noise_threshold: Optional[float] = None
) -> Dict[Tuple[int, ...], Set[int]]:
    if len(log) == 0:
        print("Error: Event log is empty")
        return {}
    activity_counts = {}
    for trace in log:
        for event in trace:
            activity = event['concept:name']
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
    discovery_parameters = {}
    if noise_threshold is not None:
        discovery_parameters["noise_threshold"] = noise_threshold

    try:
        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log, **discovery_parameters)
        labeled_transitions = sum(1 for t in net.transitions if t.label is not None)
        silent_transitions = len(net.transitions) - labeled_transitions
        rules = extract_rules_from_model(net, initial_marking, final_marking, activity_to_index)

        if rules:
            if len(rules) >= 3:
                return rules
        backup_rules = try_backup_strategies(log, activity_to_index)

        if backup_rules and len(backup_rules) > len(rules):
            return backup_rules
        elif rules:
            return rules
        else:
            return extract_simple_process_rules_fallback(log, activity_to_index)

    except Exception as e:
        return extract_simple_process_rules_fallback(log, activity_to_index)

def extract_rules_from_model(net, initial_marking, final_marking, activity_to_index):
    try:
        import threading
        result, exc = [None], [None]
        def run():
            try:
                result[0] = pm4py.discovery.discover_footprints(net, initial_marking, final_marking)
            except Exception as e:
                exc[0] = e

        t = threading.Thread(target=run)
        t.daemon = True
        t.start()
        t.join(10)

        if t.is_alive():
            return extract_rules_from_petri_net_structure(net, activity_to_index)
        if exc[0]:
            return extract_rules_from_petri_net_structure(net, activity_to_index)
        footprints = result[0]

        relation_types = ['causality', 'sequence', 'parallel', 'choice']

        for relation_type in relation_types:
            relations = footprints.get(relation_type, [])
            if relations:
                rules = build_rules_from_relations(relations, activity_to_index)
                if rules:
                    return rules
        return extract_rules_from_petri_net_structure(net, activity_to_index)

    except Exception as e:
        return {}

def extract_rules_from_petri_net_structure(net, activity_to_index):
    rules = {}

    try:
        for transition in net.transitions:
            if transition.label is None:
                continue
            if transition.label not in activity_to_index:
                continue
            current_idx = activity_to_index[transition.label]
            for arc in net.arcs:
                if arc.source == transition:
                    place = arc.target
                    for next_arc in net.arcs:
                        if next_arc.source == place and next_arc.target.label is not None:
                            next_activity = next_arc.target.label
                            if next_activity in activity_to_index:
                                next_idx = activity_to_index[next_activity]

                                pre_tuple = (current_idx,)
                                if pre_tuple not in rules:
                                    rules[pre_tuple] = set()
                                rules[pre_tuple].add(next_idx)

        return rules

    except Exception as e:
        return {}


def try_backup_strategies(log, activity_to_index, timeout_seconds=15):
    import threading
    strategies = [
        ("Alpha Miner", lambda: pm4py.discover_petri_net_alpha(log)),
        ("Heuristics Miner", lambda: pm4py.discover_petri_net_heuristics(
        log,
        dependency_threshold=0.9,
        and_threshold=0.6,
        min_act_count=2,
        min_dfg_occurrences=5
        ))
    ]

    best_rules = {}

    for strategy_name, discovery_func in strategies:
        result = [None]
        exception = [None]

        def run_discovery():
            try:
                result[0] = discovery_func()
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=run_discovery)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)

        if thread.is_alive():
            continue
        elif exception[0]:
            continue
        elif result[0] is None:
            continue

        try:
            net, initial_marking, final_marking = result[0]
            labeled_transitions = sum(1 for t in net.transitions if t.label is not None)
            silent_transitions = len(net.transitions) - labeled_transitions
            rules = extract_rules_from_model(net, initial_marking, final_marking, activity_to_index)

            if rules and len(rules) > len(best_rules):
                best_rules = rules

        except Exception as e:
            continue

    return best_rules

def extract_simple_process_rules_fallback(log, activity_to_index, min_support=2):
    total_traces = len(log)
    total_events = sum(len(trace) for trace in log)
    unique_activities = len(activity_to_index)
    if total_traces < 100:
        min_support = max(1, min_support // 2)
    elif total_traces > 1000:
        min_support = max(min_support, int(total_traces * 0.01))

    transitions = {}

    for trace in log:
        for i in range(len(trace) - 1):
            current = trace[i]['concept:name']
            next_act = trace[i + 1]['concept:name']
            key = (current, next_act)
            transitions[key] = transitions.get(key, 0) + 1

    for (from_act, to_act), count in sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]:
        percentage = (count / sum(transitions.values())) * 100 if transitions else 0
    rules = {}

    for (from_act, to_act), count in transitions.items():
        if count >= min_support and from_act in activity_to_index and to_act in activity_to_index:
            from_idx = activity_to_index[from_act]
            to_idx = activity_to_index[to_act]

            pre_tuple = (from_idx,)
            if pre_tuple not in rules:
                rules[pre_tuple] = set()
            rules[pre_tuple].add(to_idx)

    if len(rules) < 3 and min_support > 1:
        return extract_simple_process_rules_fallback(log, activity_to_index, min_support=1)
    return rules

def build_rules_from_relations(relations, activity_to_index):
    rules = {}

    for preceding_activity, subsequent_activity in relations:
        if preceding_activity in activity_to_index and subsequent_activity in activity_to_index:
            pre_idx = activity_to_index[preceding_activity]
            sub_idx = activity_to_index[subsequent_activity]

            pre_tuple = (pre_idx,)
            if pre_tuple not in rules:
                rules[pre_tuple] = set()
            rules[pre_tuple].add(sub_idx)

    return rules

def create_global_hyperedge_list_from_causal_rules(
        causal_rules
) -> List[Tuple[int, ...]]:
    global_hyperedge_list = []


    for prerequisites_tuple, consequences_set in causal_rules.items():
        if not prerequisites_tuple or not consequences_set:
            continue

        for consequence_node in consequences_set:
            hyperedge = prerequisites_tuple + (consequence_node,)
            global_hyperedge_list.append(hyperedge)

    return global_hyperedge_list

def rules_to_hyperedges(multi_rules, min_size=3):

    hyperedges = []
    for antecedent_tuple, consequents in multi_rules.items():
        for y in consequents:
            he = tuple(sorted(set(antecedent_tuple) | {y}))
            if len(he) >= min_size:
                hyperedges.append(he)
    hyperedges = sorted(set(hyperedges))
    return hyperedges

def mine_multivariate_causal_rules_by_windows(
    log,
    activity_to_index,
    base_pairwise_rules=None,
    window_size=5,
    max_antecedent_size=3,
    min_support=10,
    min_confidence=0.3
):

    allowed_pair = None
    if base_pairwise_rules:
        allowed_pair = set()
        for pre_tuple, ys in base_pairwise_rules.items():
            if len(pre_tuple) == 1:
                a = pre_tuple[0]
                for y in ys:
                    allowed_pair.add((a, y))
    count_y = Counter()
    support_rule = Counter()

    for trace in log:
        labels = [e["concept:name"] for e in trace]
        idx_seq = [activity_to_index[l] for l in labels if l in activity_to_index]

        for j, y in enumerate(idx_seq):
            count_y[y] += 1
            start = max(0, j - window_size)
            window = idx_seq[start:j]
            if not window:
                continue
            cand_set = set(window)
            if allowed_pair is not None:
                cand_set = {a for a in cand_set if (a, y) in allowed_pair}
            for k in range(2, min(max_antecedent_size, len(cand_set)) + 1):
                for S in combinations(sorted(cand_set), k):
                    support_rule[(S, y)] += 1

    multi_rules = defaultdict(set)
    for (S, y), sup in support_rule.items():
        conf = sup / count_y[y] if count_y[y] > 0 else 0.0
        if sup >= min_support and conf >= min_confidence:
            multi_rules[S].add(y)

    return dict(multi_rules)

from collections import Counter, defaultdict
from itertools import combinations

def mine_multivariate_rules_apriori(
    log,
    activity_to_index,
    base_pairwise_rules=None,
    window_size=5,
    max_antecedent_size=3,
    min_support=10,
    min_confidence=0.3,
    use_pairwise_prior=True
):

    allowed_pair = None
    if use_pairwise_prior and base_pairwise_rules:
        allowed_pair = set()
        for pre_tuple, ys in base_pairwise_rules.items():
            if len(pre_tuple) == 1:
                a = pre_tuple[0]
                for y in ys:
                    allowed_pair.add((a, y))

    count_y = Counter()
    support_rule = Counter()
    freq_itemsets = dict()
    single_count = Counter()
    for trace in log:
        labels = [e["concept:name"] for e in trace]
        idx_seq = [activity_to_index[l] for l in labels if l in activity_to_index]
        for j, y in enumerate(idx_seq):
            count_y[y] += 1
            start = max(0, j - window_size)
            window = idx_seq[start:j]
            if not window:
                continue
            unique_window = set(window)
            for a in unique_window:
                single_count[a] += 1

    freq_itemsets[1] = set([ (a,) for a, c in single_count.items() if c >= min_support ])
    if not freq_itemsets[1]:
        return {}

    max_k = min(max_antecedent_size, len(freq_itemsets[1]))
    for k in range(2, max_k + 1):
        prev_freq = sorted(freq_itemsets[k-1])
        cand_k = set()
        prev_len = len(prev_freq)
        for i in range(prev_len):
            for j in range(i+1, prev_len):
                a = prev_freq[i]
                b = prev_freq[j]
                if a[:-1] == b[:-1]:
                    candidate = tuple(sorted(set(a) | set(b)))
                    if len(candidate) == k:
                        all_subsets_freq = True
                        for subset in combinations(candidate, k-1):
                            if tuple(sorted(subset)) not in freq_itemsets[k-1]:
                                all_subsets_freq = False
                                break
                        if all_subsets_freq:
                            cand_k.add(candidate)
        if not cand_k:
            break
        cand_count = Counter()
        for trace in log:
            labels = [e["concept:name"] for e in trace]
            idx_seq = [activity_to_index[l] for l in labels if l in activity_to_index]
            for j, y in enumerate(idx_seq):
                start = max(0, j - window_size)
                window = idx_seq[start:j]
                if not window:
                    continue
                unique_window = set(window)
                for cand in cand_k:
                    skip = False
                    for a in cand:
                        if a not in unique_window:
                            skip = True
                            break
                    if not skip:
                        cand_count[cand] += 1

        freq_k = set([c for c, cnt in cand_count.items() if cnt >= min_support])
        if not freq_k:
            break
        freq_itemsets[k] = freq_k

    for trace in log:
        labels = [e["concept:name"] for e in trace]
        idx_seq = [activity_to_index[l] for l in labels if l in activity_to_index]
        for j, y in enumerate(idx_seq):
            start = max(0, j - window_size)
            window = idx_seq[start:j]
            if not window:
                continue
            unique_window = set(window)

            for k, freq_set in freq_itemsets.items():
                if k == 1:
                    for (a,) in freq_set:
                        if a in unique_window:
                            support_rule[((a,), y)] += 1
                else:
                    for S in freq_set:
                        include = True
                        for a in S:
                            if a not in unique_window:
                                include = False
                                break
                        if include:
                            if use_pairwise_prior and allowed_pair is not None:
                                pair_ok = False
                                for a in S:
                                    if (a, y) in allowed_pair:
                                        pair_ok = True
                                        break
                                if not pair_ok:
                                    continue
                            support_rule[(S, y)] += 1

    multi_rules = defaultdict(set)
    for (S, y), sup in support_rule.items():
        if sup >= min_support:
            conf = sup / (count_y[y] + 1e-12)
            if conf >= min_confidence:
                multi_rules[tuple(sorted(S))].add(y)

    return dict(multi_rules)

if __name__ == '__main__':

    list_eventlog = [
                      #'BPI2020_Domestic',
                      #'BPI2020_Prepaid',
                      #'BPI2020_Request',
                      #'env_permit',
                      #'bpi13_closed_problems',
                      'receipt'
                     ]


    for eventlog_name in list_eventlog:
        fold = 0
        base_dir = "data"
        csv_paths = {
            "train_csv": os.path.join(base_dir, eventlog_name,
                                      f"{eventlog_name}_kfoldcv_{fold}_train.csv"),
            "valid_csv": os.path.join(base_dir, eventlog_name,
                                      f"{eventlog_name}_kfoldcv_{fold}_valid.csv"),
            "test_csv": os.path.join(base_dir, eventlog_name,
                                     f"{eventlog_name}_kfoldcv_{fold}_test.csv")
        }
        train_log = load_and_convert_csv_for_all_data(csv_paths["train_csv"])
        valid_log = load_and_convert_csv_for_all_data(csv_paths["valid_csv"])
        test_log = load_and_convert_csv_for_all_data(csv_paths["test_csv"])

        merge_log = merge_two_logs(train_log, valid_log)
        log = merge_two_logs(merge_log,test_log)

        mapping_path = os.path.join(
            base_dir, eventlog_name,
            f"{eventlog_name}_y_{fold}_ActivityMap.pickle"
        )

        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Activity mapping file does not exist. Please check the path:\n{mapping_path}")

        with open(mapping_path, "rb") as f:
            activity_mapping = pickle.load(f)
        print(activity_mapping)

        causal_rules = discover_and_index_causal_rulesV2(log, activity_mapping, 0.3)
        causal_hyperedge_list = create_global_hyperedge_list_from_causal_rules(causal_rules)


        multi_rules = mine_multivariate_rules_apriori(
            log=log,
            activity_to_index=activity_mapping,
            base_pairwise_rules=causal_rules,
            window_size=5,
            max_antecedent_size=3,
            min_support=10,
            min_confidence=0.3,
            use_pairwise_prior=True
        )

        multi_hyperedges = rules_to_hyperedges(multi_rules)
        if multi_hyperedges==[]:
            multi_hyperedges = causal_hyperedge_list

        print(f"print {eventlog_name} hyperedge list")
        print(multi_hyperedges)
        hyperedge_path = os.path.join(
            base_dir, eventlog_name,
            f"{eventlog_name}_y_CE_hyperedge.pickle"
        )
        print(causal_hyperedge_list)
        with open(hyperedge_path, "wb") as f:
            pickle.dump(multi_hyperedges, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✅ hyperedge list have saved to: {hyperedge_path}")



