"""Verify frozen arm-annotation flags and evidence; standard library only."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def yes(value):
    return str(value).lower() == 'true'


def key(row):
    return row['arm'], row['gene']


def main():
    manifest = json.loads((ROOT / 'FILE_MANIFEST.json').read_text(encoding='utf-8'))
    for entry in manifest:
        path = ROOT / entry['file']
        assert path.is_file() and path.stat().st_size == entry['bytes'], entry['file']
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == entry['sha256'], entry['file']
    flags = rows('arm_annotation_flags.csv')
    audit = rows('audit/label_classes.csv')
    selected = [r for r in audit if yes(r['suspect'])]
    assert len(flags) == len(selected) == 137
    assert len({key(r) for r in flags}) == 137
    assert {key(r) for r in flags} == {key(r) for r in selected}
    assert len({r['arm'] for r in flags}) == 78
    assert Counter(r['set'] for r in flags) == {'val': 110, 'test': 27}
    assert Counter(r['own_site_class'] for r in flags) == {'none': 104, '6mer': 33}
    assert all((r['validation_fold'] in {'1', '2', '3', '4', '5'}) if r['set'] == 'val'
               else r['validation_fold'] == '' for r in flags)
    features = {key(r): r for r in rows('evidence/seed_site_features.csv')}
    clash = rows('evidence/clash_flagged_pairs.csv')
    assert all(float(r['WT_Detected_8']) >= 1 for r in clash)
    binding = {(r['arm'], r['gene_name']) for r in clash}
    detected = {r['arm'] for r in rows('evidence/clash_detected_mirnas.csv')}
    audit_by_key = {key(r): r for r in selected}
    for row in flags:
        own_key = key(row)
        other_key = row['other_arm'], row['gene']
        own, other = features[own_key], features[other_key]
        canonical = lambda f: any(float(f[c]) > 0 for c in ['log1p_8mer', 'log1p_7mer_m8', 'log1p_7mer_A1'])
        assert not canonical(own) and canonical(other)
        assert float(own['sequence_available']) == float(other['sequence_available']) == 1
        assert row['own_site_class'] == own['site_class'] and row['other_site_class'] == other['site_class']
        assert row['role'] == 'minor_arm' and yes(row['shared']) and yes(row['suspect'])
        assert not yes(row['own_site']) and yes(row['other_site'])
        assert yes(row['clash_own']) == (own_key in binding)
        assert yes(row['clash_other']) == (other_key in binding)
        assert yes(row['clash_measurable']) == (row['arm'] in detected and row['other_arm'] in detected)
        original = audit_by_key[own_key]
        assert original['role'] == '次要臂' and row['other_arm'] == original['other_arm']
        for field in ['shared', 'own_site', 'other_site', 'clash_own', 'clash_other', 'clash_measurable']:
            assert yes(row[field]) == yes(original[field])
    diagnostic = rows('diagnostics/flagged_label_rows.csv')
    assert {(r['mirna'].lower(), r['gene']) for r in diagnostic} == {key(r) for r in flags}
    print(json.dumps({'ok': True, 'files_verified': len(manifest), 'unique_flags': len(flags),
                      'miRNAs': 78, 'test_flags': 27, 'sequence_and_clash_checks_passed': True,
                      'training_executed': False, 'inference_executed': False}, indent=2))


if __name__ == '__main__':
    main()
