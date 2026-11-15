"""Bounded causal graph comparisons; all attribution is conditional on graph assumptions."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import pickle
import platform
import shutil
import time
import traceback

import numpy as np
import pandas as pd
from analysis import causal_common as cc


@dataclass(frozen=True)
class Spec:
    model: str = 'pcmci'
    tau: int = 4
    release: bool = False
    window: int | None = None
    spline: bool = False
    assumptions: bool = True
    matched: bool = False


def contiguous(X):
    """Longest complete calendar-contiguous segment; never stitch missing weeks."""
    valid = X.notna().all(axis=1)
    group = ((~valid) | (X.index.to_series().diff() != pd.Timedelta(weeks=1))).cumsum()
    groups = [v for _, v in X[valid].groupby(group[valid])]
    if not groups:
        raise ValueError('No complete contiguous training segment')
    return max(groups, key=len)


def zero_before_start(X):
    """Weeks before a driver's first observation become 0 (the z-score mean: no information); later gaps stay missing."""
    return X.mask(X.ffill().isna() & (X.columns != 'price'), 0.)


def parents(graph, columns):
    return {(columns[i], int(t)) for i, t in zip(*np.nonzero(graph[:, 0, :] == '-->'))}


def discover(X, sink, spec):
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    if spec.model == 'varlingam':
        from lingam import VARLiNGAM
        model = VARLiNGAM(lags=spec.tau, criterion=None, prune=True, random_state=0).fit(X.values)
        g = np.full((X.shape[1], X.shape[1], spec.tau + 1), '', dtype='<U3')
        for t, B in enumerate(model.adjacency_matrices_):
            for j, i in zip(*np.nonzero(B)):
                g[i, j, t] = '-->'
        return g
    frame = pp.DataFrame(X.fillna(cc.MISSING).values, missing_flag=cc.MISSING)
    opts = dict(tau_max=spec.tau, pc_alpha=.05)
    if spec.assumptions:
        opts['link_assumptions'] = cc.link_assumptions(sink, pag=spec.model == 'lpcmci', tau_max=spec.tau)
    if spec.model == 'lpcmci':
        from tigramite.lpcmci import LPCMCI
        return LPCMCI(frame, ParCorr(significance='analytic'), verbosity=0).run_lpcmci(max_p_global=3, **opts)['graph']
    if spec.model == 'robust':
        from tigramite.independence_tests.robust_parcorr import RobustParCorr
        test = RobustParCorr(significance='analytic')
    elif spec.model == 'gpdc':
        from tigramite.independence_tests.gpdc import GPDC
        test = GPDC(significance='analytic', seed=0)
    else:
        test = ParCorr(significance='analytic')
    return PCMCI(frame, test, verbosity=0).run_pcmciplus(tau_min=0, **opts)['graph']


def contributions(X, terms, fit, spline=False, details=False):
    terms = sorted(terms)
    out = pd.DataFrame(0., index=X.index, columns=X.columns.drop('price'))
    fit = np.asarray(fit) & X.price.notna().values
    if not terms:
        mean = float(X.price.loc[fit].mean())
        prediction = pd.Series(mean, index=X.index, name='fitted_standardized_return')
        return (out, prediction, dict(terms=[], intercept=mean)) if details else out
    D = pd.DataFrame({str(i): X[c].shift(t) for i, (c, t) in enumerate(terms)}).fillna(0)
    fit = np.asarray(fit) & X.price.notna().values
    if spline:
        from sklearn.preprocessing import SplineTransformer
        from sklearn.linear_model import Ridge
        trans = SplineTransformer(n_knots=3, degree=3, include_bias=False)
        trans.fit(D.loc[fit])
        Z = trans.transform(D)
        model = Ridge(alpha=1).fit(Z[fit], X.price.loc[fit])
        prediction = model.predict(Z)
        fitted = dict(terms=terms, transformer=trans, estimator=model)
        Z = (Z - Z[fit].mean(axis=0)) * model.coef_
        width = Z.shape[1] // len(terms)
        values = np.column_stack([Z[:, i*width:(i+1)*width].sum(axis=1) for i in range(len(terms))])
    else:
        coef = np.linalg.lstsq(np.column_stack([np.ones(fit.sum()), D.values[fit]]), X.price.values[fit], rcond=None)[0]
        beta = coef[1:]
        prediction = coef[0] + D.values @ beta
        fitted = dict(terms=terms, intercept=coef[0], coefficients=beta)
        values = (D.values - D.values[fit].mean(axis=0)) * beta
    for i, (c, _) in enumerate(terms):
        if c != 'price':
            out[c] += values[:, i]
    prediction = pd.Series(prediction, index=X.index, name='fitted_standardized_return')
    return (out, prediction, fitted) if details else out


def job(series, phase, spec, graph_from=None):
    end = '2023-01-01' if phase == 'selection' else '2025-01-01'
    start = None if spec.window is None else pd.Timestamp(end) - pd.Timedelta(weeks=spec.window)
    X, info = cc.make_panel(series, fit_end=end, train_start=start)
    fit = (X.index < end) & (True if start is None else X.index >= start)
    if spec.release:
        X = cc.release_panel(cc.event_panel(series, X, info, fit_end=end, train_start=start), 'surprise', end)
    training = X.loc[fit]
    if spec.matched or spec.model == 'varlingam':
        training = contiguous(zero_before_start(training))  # late-starting drivers would otherwise cut the window to their overlap
        if len(training) < max(52, 2 * X.shape[1] * (spec.tau + 1)):
            raise ValueError('Insufficient contiguous complete training weeks')
        fit = X.index.isin(training.index)
    g = graph_from["graph"] if graph_from is not None else discover(training, cc.sinks(X, info), spec)
    terms = parents(g, X.columns)
    c, prediction, fitted = contributions(X, terms, fit, spec.spline, details=True)
    key = cc.load_key(series)
    key = key.loc['2023-07-01':'2024-12-31'] if phase == 'selection' else key.loc['2025-01-01':]
    universe, _ = cc.make_panel(series)
    c = c.reindex(columns=universe.columns.drop('price'), fill_value=0.)
    scores = cc.score_weeks(c, universe, key)
    hits = cc.week_hits(c, universe, key, 3)
    return dict(scores=scores, hits=hits, contributions=c.loc[key.index], graph=g,
                predictions=pd.concat([prediction, X.price.rename('actual_standardized_return')], axis=1).loc[key.index],
                fitted_model=fitted,
                parents=sorted(terms), columns=list(X.columns), train_start=str(training.index.min()),
                train_end=str(training.index.max()), train_rows=len(training),
                assumed_parents=[(v,t) for v,t in sorted(terms) if t == 0 and spec.assumptions and spec.model != 'varlingam' and cc.sinks(X,info)[X.columns.get_loc(v)]],
                candidate_count=len({v for v,t in terms if v != 'price'}))


def fingerprint(series=None, models=None):
    files = [Path(__file__), Path(cc.__file__), cc.PANEL, cc.KEY, cc.RELEASE_FILE,
             cc.ROOT/'data/01_raw/exogenous.parquet', cc.ROOT/'uv.lock']
    h = hashlib.sha256(json.dumps(dict(series=series or list(cc.SERIES),models=models),sort_keys=True).encode())
    for p in files:
        h.update(str(p).encode()); h.update(p.read_bytes())
    h.update(platform.python_version().encode())
    for name in ('numpy','pandas','tigramite','lingam','scikit-learn','dcor'):
        try: h.update((name + importlib.metadata.version(name)).encode())
        except importlib.metadata.PackageNotFoundError: h.update((name + ':missing').encode())
    return h.hexdigest()[:20]


def cached(path):
    try:
        manifest = json.loads((path/'manifest.json').read_text())
        required = {'result.pkl','sha256','contributions.parquet','hits.parquet','ranking.parquet','parents.json','spec.json','predictions.parquet','model.pkl'}
        if set(manifest) != required: return None
        if any(hashlib.sha256((path/name).read_bytes()).hexdigest() != digest for name,digest in manifest.items()): return None
        data = (path/'result.pkl').read_bytes()
        if hashlib.sha256(data).hexdigest() != (path/'sha256').read_text(): return None
        return pickle.loads(data)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError):
        return None


def worker(path, series, phase, spec, graph_from=None):
    try:
        began=time.monotonic()
        result = job(series, phase, spec, graph_from)
        result['compute_seconds']=time.monotonic()-began
        payload = pickle.dumps(result)
        (path/'result.pkl').write_bytes(payload)
        (path/'sha256').write_text(hashlib.sha256(payload).hexdigest())
        result['contributions'].to_parquet(path/'contributions.parquet')
        result['hits'].rename('hit').to_frame().to_parquet(path/'hits.parquet')
        result['predictions'].to_parquet(path/'predictions.parquet')
        (path/'model.pkl').write_bytes(pickle.dumps(result['fitted_model']))
        ranking=[]
        for week,row in result['contributions'].iterrows():
            selected=row.loc[row.abs().sort_values(ascending=False,kind='stable').index]
            for rank,(variable,value) in enumerate(selected[selected!=0].head(3).items(),1):
                ranking.append(dict(week=week,rank=rank,variable=variable,contribution=float(value)))
        pd.DataFrame(ranking,columns=['week','rank','variable','contribution']).to_parquet(path/'ranking.parquet',index=False)
        (path/'parents.json').write_text(json.dumps(result['parents']))
        files=['result.pkl','sha256','contributions.parquet','hits.parquet','ranking.parquet','parents.json','spec.json','predictions.parquet','model.pkl']
        (path/'manifest.json').write_text(json.dumps({name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in files}))
    except Exception:
        (path/'error.txt').write_text(traceback.format_exc())


def paired_ci(a, b, n=2000):
    """Macro mean difference, resampling shared four-calendar-week blocks."""
    ds = []
    for s in a:
        if not a[s].index.equals(b[s].index): raise ValueError('Evaluation weeks differ')
        ds.append((a[s].astype(float)-b[s].astype(float)).rename(s))
    frame = pd.concat(ds, axis=1).sort_index()
    blocks = ((frame.index-frame.index.min()).days // 28)
    groups = [v for _, v in frame.groupby(blocks)]
    rng = np.random.default_rng(0)
    draws = [pd.concat([groups[i] for i in rng.integers(0,len(groups),len(groups))]).mean().mean() for _ in range(n)]
    return [float(frame.mean().mean()), *map(float,np.nanpercentile(draws,[2.5,97.5]))]


def _run(output, hours=12, fit_minutes=45, series=None, models=None):
    series = series or list(cc.SERIES)
    output = Path(output)/fingerprint(series,models); output.mkdir(parents=True, exist_ok=True)
    source = output/'source'; source.mkdir(exist_ok=True)
    for path in (Path(__file__), Path(cc.__file__), cc.ROOT/'tools/run_causal_study.py', cc.ROOT/'pyproject.toml', cc.ROOT/'uv.lock'):
        shutil.copy2(path, source/path.name)
    inputs = [cc.PANEL, cc.KEY, cc.RELEASE_FILE, cc.ROOT/'data/01_raw/exogenous.parquet']
    (output/'inputs.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2))
    (output/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}),indent=2))
    deadline = time.monotonic()+hours*3600
    records, results, specs = [], {}, {}
    def execute(name, spec, phase, parent='', axis='model'):
        specs[name] = spec
        for s in series:
            path = output/phase/name/s; path.mkdir(parents=True, exist_ok=True)
            (path/'spec.json').write_text(json.dumps(asdict(spec), indent=2))
            result = cached(path); began = time.monotonic(); status = 'cached' if result else 'pending'
            if result is None:
                (output/'status.json').write_text(json.dumps(dict(status='running',current_job=dict(name=name,phase=phase,series=s))))
                remaining = deadline-time.monotonic()
                if remaining <= 0: status = 'budget_exhausted'
                else:
                    proc = mp.get_context('spawn').Process(target=worker,args=(path,s,phase,spec,results.get((phase,parent,s)) if spec.spline and parent in specs and replace(spec,spline=False)==replace(specs[parent],spline=False) else None)); proc.start()
                    proc.join(min(fit_minutes*60, remaining))
                    if proc.is_alive(): proc.terminate(); proc.join(); status = 'timeout'
                    else: result = cached(path); status = 'complete' if result else 'failed'
            row = dict(name=name, phase=phase, series=s, parent=parent, axis=axis,status=status,seconds=time.monotonic()-began)
            if result:
                results[phase,name,s] = result
                row['seconds']=result['compute_seconds']
                row.update(result['scores']); row['candidate_count'] = result['candidate_count']
            records.append(row)
            pd.DataFrame(records).to_csv(output/'summary.csv',index=False)
            (output/'status.json').write_text(json.dumps(dict(status='running',last_job=row),default=str))
            print(f'{phase} {name} {s}: {status}',flush=True)
    def complete(name,phase='selection'): return all((phase,name,s) in results for s in series)
    def score(name): return np.mean([results['selection',name,s]['scores']['top3'] for s in series])
    def choose(names):
        eligible = [n for n in names if complete(n)]
        return min(eligible,key=lambda n:(-score(n),n!='baseline',sum(r['seconds'] for r in records if r['name']==n))) if eligible else None
    names=[]
    for model in [m for m in (models or ['pcmci','robust','gpdc','lpcmci']) if m != 'varlingam']:
        name = 'baseline' if model=='pcmci' else model
        execute(name,Spec(model=model),'selection'); names.append(name)
    if models is None or 'varlingam' in models:
        execute('matched_pcmci',Spec(matched=True),'selection',axis='matched_window')
        execute('varlingam',Spec(model='varlingam',matched=True,assumptions=False),'selection',parent='matched_pcmci',axis='model_and_direction')
    winner=choose(names); selected=winner
    if winner:
        variants={'release':dict(release=True),'tau8':dict(tau=8),'window260':dict(window=260),'spline':dict(spline=True)}
        ablations=[]
        for axis,changes in variants.items():
            name=winner+'_'+axis; execute(name,replace(specs[winner],**changes),'selection',winner,axis); ablations.append(name)
        selected=choose([winner,*ablations])
        for name in dict.fromkeys(['baseline',winner,selected]):
            if name in specs: execute(name,specs[name],'confirmation',parent=winner if name==selected and name!=winner else ('' if name=='baseline' else 'baseline'))
        execute(selected+'_no_direction',replace(specs[selected],assumptions=False),'confirmation',selected,'direction_assumption')
    if models is None or "varlingam" in models:
        execute("matched_pcmci",Spec(matched=True),"confirmation",axis="matched_window")
        execute("varlingam",Spec(model="varlingam",matched=True,assumptions=False),"confirmation",parent="matched_pcmci",axis="model_and_direction")
    comparisons=[]
    for phase in ('selection','confirmation'):
        for name in dict.fromkeys(r['name'] for r in records if r['phase']==phase):
            parent=next(r['parent'] for r in records if r['phase']==phase and r['name']==name) or 'baseline'
            controls = [parent]
            if name not in ('varlingam','matched_pcmci') and parent != 'baseline':
                controls.append('baseline')
            for control in controls:
                if name!=control and complete(name,phase) and complete(control,phase):
                    ci=paired_ci({s:results[phase,name,s]['hits'] for s in series},{s:results[phase,control,s]['hits'] for s in series})
                    comparisons.append(dict(phase=phase,name=name,parent=control,delta=ci[0],low=ci[1],high=ci[2],improved=phase=='confirmation' and ci[1]>0))
    pd.DataFrame(comparisons).to_csv(output/'comparisons.csv',index=False)
    (output/'selection.json').write_text(json.dumps(dict(winner=winner,selected=selected,specs={k:asdict(v) for k,v in specs.items()}),indent=2))
    (output/'report.md').write_text('# Causal attribution study\n\nHistorical exploratory periods; no unseen holdout. Attribution is conditional on graph assumptions.\n\nVARLiNGAM is a separate matched-window comparison and is not ranked against full-window models. Failed or timed-out fits are incomplete, not zero scores. Predictions are standardized weekly returns, not price forecasts.\n\n'+f'Selection winner: {winner}; selected ablation: {selected}.\n\n```\n'+pd.DataFrame(records).to_string(index=False)+'\n```\n\n```\n'+pd.DataFrame(comparisons).to_string(index=False)+'\n```\n')
    return output


def run(output, hours=12, fit_minutes=45, series=None, models=None):
    import fcntl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A causal study already runs in this output root')
        path = output/fingerprint(series,models)
        (output/'latest.json').write_text(json.dumps({'path':str(path.resolve())}))
        path.mkdir(parents=True, exist_ok=True)
        (path/'status.json').write_text(json.dumps({'status':'running'}))
        try:
            result = _run(output,hours,fit_minutes,series,models)
            summary = pd.read_csv(result/'summary.csv')
            state = 'complete' if summary.status.isin(['complete','cached']).all() else 'incomplete'
            (path/'status.json').write_text(json.dumps({'status':state}))
            return result
        except BaseException:
            (path/'status.json').write_text(json.dumps({'status':'failed','error':traceback.format_exc()}))
            raise
