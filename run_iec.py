from __future__ import annotations
import argparse, hashlib, json, os, platform
from pathlib import Path
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from config import CLASS_NAMES, METHOD_NAMES, OUTER_REPEATS, OUTER_SPLITS, OUTPUT_DIR, PROJECT_ID, RANDOM_SEED, RUN_MODE, protocol_payload
from data import load_dataset
from evaluation import probability_metrics, summarize_internal
from nested_tuning import fit_select_predict
from reporting import (
    build_leakage_audit,
    parameter_frequencies,
    write_fold_metrics_csv,
    write_main_tables_csv,
    write_predictions_csv,
    write_selected_parameters_csv,
)

def dump(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def rows(ids, y, variants):
    out=[]
    for method in METHOD_NAMES:
        p=variants[method]; pred=np.argmax(p, axis=1)
        for i, rid in enumerate(ids):
            out.append({'record_id':str(rid),'true_label':CLASS_NAMES[int(y[i])],'method':method,'predicted_label':CLASS_NAMES[int(pred[i])], **{f'p_{name}':float(p[i,j]) for j,name in enumerate(CLASS_NAMES)}})
    return out

def main():
    ap=argparse.ArgumentParser(description='IEC 167-only formal E1 experiment')
    ap.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    ap.add_argument('--validate-only', action='store_true')
    args=ap.parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    payload=protocol_payload(); sha=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dataset, source_audit=load_dataset(); dump(source_audit,out/'source_boundary_audit.json')
    if args.validate_only:
        print(f'IEC-only validation passed: 167 records; E1 only; mode={RUN_MODE}'); return
    experiment_id=f'E1_IEC_INTERNAL_{OUTER_REPEATS}X{OUTER_SPLITS}'
    splitter=RepeatedStratifiedKFold(n_splits=OUTER_SPLITS,n_repeats=OUTER_REPEATS,random_state=RANDOM_SEED); folds=[]; ckpt=out/'checkpoints'/experiment_id
    for fold,(train,test) in enumerate(splitter.split(dataset.gases,dataset.y)):
        path=ckpt/f'fold_{fold:02d}.json'
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            if (
                saved.get('fold') == fold
                and saved.get('protocol_sha256') == sha
            ):
                folds.append(saved)
                continue
        print(f"fold {fold + 1}/{OUTER_SPLITS * OUTER_REPEATS}")
        variants,audit=fit_select_predict(dataset.gases.iloc[train],dataset.y[train],dataset.record_ids[train],dataset.gases.iloc[test],dataset.record_ids[test],seed=RANDOM_SEED+fold*100003,progress=print)
        metrics={m:probability_metrics(dataset.y[test],variants[m],len(CLASS_NAMES)) for m in METHOD_NAMES}
        value={'protocol_sha256':sha,'fold':fold,'repeat':fold//OUTER_SPLITS,'fold_within_repeat':fold%OUTER_SPLITS,'train_indices':train.tolist(),'test_indices':test.tolist(),'train_record_ids':[str(x) for x in dataset.record_ids[train]],'test_record_ids':[str(x) for x in dataset.record_ids[test]],'train_test_overlap_count':0,'outer_test_labels_used_for_selection':False,'metrics':metrics,'predictions':rows(dataset.record_ids[test],dataset.y[test],variants),'audit':audit}
        dump(value,path); folds.append(value)
    experiment={'id':experiment_id,'title':f'IEC internal {OUTER_REPEATS}x{OUTER_SPLITS} nested stratified cross-validation','kind':f'internal_{OUTER_REPEATS}x{OUTER_SPLITS}_cv','source':'IEC','samples':167,'outer_fold_count':len(folds),'folds':folds,'summary':summarize_internal([f['metrics'] for f in folds],METHOD_NAMES)}
    status = (
        'QUICK_DIAGNOSTIC_COMPLETE_NO_POST_HOC_SELECTION'
        if RUN_MODE == 'quick'
        else 'FORMAL_COMPLETE_NO_POST_HOC_SELECTION'
    )
    result={'project_id':PROJECT_ID,'status':status,'run_mode':RUN_MODE,'protocol_sha256':sha,'protocol':payload,'source_boundary_audit':source_audit,'software':{'python':platform.python_version(),'numpy':np.__version__},'experiments':{experiment_id:experiment}}
    dump(result,out/'DGA_IEC167_FORMAL_RESULTS.json')
    leakage = build_leakage_audit(result)
    if not leakage['overall_pass']:
        raise RuntimeError('IEC leakage audit failed')
    dump(leakage,out/'leakage_audit.json')
    dump(parameter_frequencies(result),out/'selected_parameter_frequency.json')
    write_main_tables_csv(result,out/'DGA_IEC167_FORMAL_MAIN_TABLE.csv')
    write_fold_metrics_csv(result,out/'fold_metrics.csv')
    write_selected_parameters_csv(result,out/'selected_parameters.csv')
    write_predictions_csv(result,out/'predictions.csv')
    print(f'IEC-only E1 complete: {len(folds)} folds, 167 records, AKFMMF only')

if __name__ == '__main__': main()
