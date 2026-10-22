"""Part 1/4: Streamlit entry point, dataset upload, shared engine and staged navigation.

Run with Python 3.12 (recommended):
    python -m pip install -r requirements.txt
    python -m streamlit run racc_streamlit_app.py

Keep part_2_data.py, part_3_models.py and part_4_results.py beside this entry file.
The original notebook is not required. Run the four stages separately in the interface.
Required CSV headers: DATE,TIME,LOCATION,WEATHER,ROAD CONDITION,VEHICLE TYPE,CAUSE,SEVERITY,GENDER.
The analysis defaults and contextual date fixes target the original 2021–2025 RACC data.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

from pathlib import Path
from io import BytesIO
import hashlib, importlib.metadata, itertools, json, math, platform, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, classification_report, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, mean_absolute_error,
    mean_squared_error, r2_score, mean_poisson_deviance)
from sklearn.calibration import calibration_curve
import joblib

import tempfile
import threading
import time
import traceback
import zipfile
import streamlit as st

APP_VERSION='2.0.0'
DEFAULT_CONFIG={
    'apply_context_date_fixes':True,'drop_exact_duplicates':False,
    'include_investigated_cause':False,'assume_complete_reporting_calendar':True,
    'forecast_horizon':12,'n_permutations':999,'n_bootstrap':500,
    'n_importance_repeats':10,'csv_encoding':'utf-8-sig','csv_separator':','}
REQUIRED_COLUMNS=['DATE','TIME','LOCATION','WEATHER','ROAD CONDITION','VEHICLE TYPE','CAUSE','SEVERITY','GENDER']

@st.cache_resource(show_spinner=False)
def shared_plot_lock():
    """Serialize Matplotlib work across browser sessions."""
    return threading.RLock()

class MarkdownText(str):
    pass

MODULE_SPECS = {'00_setup': {'title': 'Setup and dataset loading',
              'function': 'module_00_load_dataset',
              'folder': '00_setup',
              'requires': []},
 '01_cleaning': {'title': 'Data cleaning and preparation',
                 'function': 'module_01_clean_data',
                 'folder': '01_data_cleaning',
                 'requires': ['00_setup']},
 '02_statistics': {'title': 'Statistical analysis',
                   'function': 'module_02_statistical_analysis',
                   'folder': '02_statistical_analysis',
                   'requires': ['01_cleaning']},
 '03_eda': {'title': 'Exploratory data analysis',
            'function': 'module_03_exploratory_analysis',
            'folder': '03_exploratory_analysis',
            'requires': ['01_cleaning']},
 '04_classification': {'title': 'Classification algorithms',
                       'function': 'module_04_classification',
                       'folder': '04_classification',
                       'requires': ['01_cleaning']},
 '05_prediction': {'title': 'Numeric prediction algorithms',
                   'function': 'module_05_numeric_prediction',
                   'folder': '05_numeric_prediction',
                   'requires': ['01_cleaning']},
 '06_forecasting': {'title': 'Forecasting algorithms',
                    'function': 'module_06_forecasting',
                    'folder': '06_forecasting',
                    'requires': ['01_cleaning']},
 '07_performance': {'title': 'Model performance evaluation',
                    'function': 'module_07_model_performance',
                    'folder': '07_model_performance',
                    'requires': ['01_cleaning',
                                 '04_classification',
                                 '05_prediction',
                                 '06_forecasting']},
 '08_features': {'title': 'Features and variable combinations',
                 'function': 'module_08_feature_analysis',
                 'folder': '08_features_and_combinations',
                 'requires': ['01_cleaning',
                              '04_classification',
                              '05_prediction',
                              '07_performance']},
 '09_exports': {'title': 'Exports and consolidated results',
                'function': 'module_09_export_results',
                'folder': '09_exports',
                'requires': ['00_setup',
                             '01_cleaning',
                             '02_statistics',
                             '03_eda',
                             '04_classification',
                             '05_prediction',
                             '06_forecasting',
                             '07_performance',
                             '08_features']}}


def binary_metrics(y,pred,prob):
    return {'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),
        'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
        'severe_precision':precision_score(y,pred,zero_division=0),
        'severe_recall':recall_score(y,pred,zero_division=0),
        'ROC_AUC':roc_auc_score(y,prob) if np.unique(y).size==2 else np.nan,
        'PR_AUC_average_precision':average_precision_score(y,prob) if np.unique(y).size==2 else np.nan,
        'Brier_score':brier_score_loss(y,prob),
        'log_loss':log_loss(y,np.column_stack([1-prob,prob]),labels=[0,1])}

def count_metrics(actual,predicted):
    a=np.asarray(actual,dtype=float);p=np.clip(np.asarray(predicted,dtype=float),0,None)
    denominator=np.abs(a).sum()
    smape_denominator=np.abs(a)+np.abs(p)
    smape_terms=np.divide(2*np.abs(a-p),smape_denominator,out=np.zeros_like(a),where=smape_denominator>0)
    return {'MAE':mean_absolute_error(a,p),'RMSE':np.sqrt(mean_squared_error(a,p)),
            'R2':r2_score(a,p) if len(a)>1 else np.nan,
            'WAPE_pct':100*np.abs(a-p).sum()/denominator if denominator>0 else np.nan,
            'sMAPE_pct':100*np.mean(smape_terms),
            'Poisson_deviance':mean_poisson_deviance(a,np.maximum(p,1e-9))}




PART_TASKS = {1: ['00_setup'], 2: ['01_cleaning', '02_statistics', '03_eda'], 3: ['04_classification', '05_prediction', '06_forecasting'], 4: ['07_performance', '08_features', '09_exports']}
MODULE_FILES = {'01_cleaning': 'part_2_data',
 '02_statistics': 'part_2_data',
 '03_eda': 'part_2_data',
 '04_classification': 'part_3_models',
 '05_prediction': 'part_3_models',
 '06_forecasting': 'part_3_models',
 '07_performance': 'part_4_results',
 '08_features': 'part_4_results',
 '09_exports': 'part_4_results'}

def create_engine(source_bytes,source_filename,config=None,plot_lock=None):
    """Create one isolated analysis state for one dataset and browser session.

    No uploaded dataset or fitted model is cached across users. Each engine owns a temporary
    output directory, its dataframes/models, and captured tables/figures for Streamlit reruns.
    """
    config={**DEFAULT_CONFIG,**(config or {})}
    source_filename=Path(str(source_filename)).name
    source_bytes=bytes(source_bytes)
    if Path(source_filename).suffix.lower()!='.csv':
        raise ValueError('Upload a CSV file (.csv).')
    if not source_bytes:
        raise ValueError('The uploaded CSV is empty.')
    owner=tempfile.TemporaryDirectory(prefix='racc_streamlit_')
    OUTPUT_ROOT=Path(owner.name)/'analysis_outputs'
    OUTPUT_ROOT.mkdir()
    LOCK=plot_lock or threading.RLock()
    STATE={}
    EVENTS={}
    ACTIVE_MODULE=None
    DURATIONS={}
    SEED=42
    CSV_ENCODING=config['csv_encoding']
    CSV_SEPARATOR=config['csv_separator']
    APPLY_CONTEXT_DATE_FIXES=config['apply_context_date_fixes']
    DROP_EXACT_DUPLICATES=config['drop_exact_duplicates']
    INCLUDE_INVESTIGATED_CAUSE=config['include_investigated_cause']
    ASSUME_COMPLETE_REPORTING_CALENDAR=config['assume_complete_reporting_calendar']
    SEVERITY_TEST_YEAR=2024
    COUNT_TEST_YEAR=2025
    FIRST_MONTH='2021-01-01'
    LAST_MONTH='2025-12-01'
    FORECAST_HORIZON=int(config['forecast_horizon'])
    N_PERMUTATIONS=int(config['n_permutations'])
    N_BOOTSTRAP=int(config['n_bootstrap'])
    N_IMPORTANCE_REPEATS=int(config['n_importance_repeats'])
    MIN_COMBINATION_RECORDS=12
    ALPHA=.05
    versions={p:importlib.metadata.version(p) for p in
              ['numpy','pandas','scipy','scikit-learn','matplotlib','seaborn','joblib','streamlit']}
    with LOCK:
        sns.set_theme(style='whitegrid',context='notebook',palette='colorblind')
        plt.rcParams.update({'figure.figsize':(10,4.5),'figure.dpi':110,'savefig.dpi':160})

    def append_event(kind,data,title=None):
        if ACTIVE_MODULE is not None:
            EVENTS.setdefault(ACTIVE_MODULE,[]).append({'kind':kind,'data':data,'title':title})

    def display(obj):
        if isinstance(obj,(pd.DataFrame,pd.Series)):
            table=obj.to_frame() if isinstance(obj,pd.Series) else obj
            append_event('table',table.copy(deep=True),table.attrs.get('_racc_table_title'))
        elif isinstance(obj,MarkdownText): append_event('markdown',str(obj))
        else: append_event('text',str(obj))

    def Markdown(value): return MarkdownText(value)

    def print(*args,sep=' ',end='\n',**kwargs):
        append_event('text',sep.join(str(x) for x in args)+end)

    def output_helpers(directory):
        directory.mkdir(parents=True,exist_ok=True)
        def save_table(table,name,index=False):
            table.to_csv(directory/f'{name}.csv',index=index)
            labeled=table.copy(deep=False)
            labeled.attrs={**table.attrs,'_racc_table_title':name}
            return labeled
        def finish_figure(name):
            plt.tight_layout()
            fig=plt.gcf()
            fig.savefig(directory/f'{name}.png',bbox_inches='tight',dpi=160)
            buffer=BytesIO()
            fig.savefig(buffer,format='png',bbox_inches='tight',dpi=110)
            append_event('image',buffer.getvalue(),name)
            plt.close('all')
        return save_table,finish_figure

    def get_selected_dataset():
        return {'name':source_filename,'content':source_bytes}

    def module_00_load_dataset(state):
        """Load and validate the CSV chosen in the upload step.

        Input: one explicitly uploaded CSV, or a CSV chosen through path mode.
        Returns: source rows, required columns, source filename/bytes and SHA-256.
        """
        OUTPUT_DIR=OUTPUT_ROOT/'00_setup'
        save_table,finish_figure=output_helpers(OUTPUT_DIR)
        selected=get_selected_dataset()
        source_bytes=selected['content']
        source_name=selected['name']
        SOURCE_SHA256=hashlib.sha256(source_bytes).hexdigest()
        try:
            raw=pd.read_csv(BytesIO(source_bytes),dtype='string',encoding=CSV_ENCODING,sep=CSV_SEPARATOR)
        except UnicodeDecodeError as exc:
            raise ValueError('CSV encoding could not be read. Update CSV_ENCODING, then rerun Module 0.B.') from exc
        except (pd.errors.EmptyDataError,pd.errors.ParserError) as exc:
            raise ValueError('The file is not a readable CSV. Check its contents and CSV_SEPARATOR.') from exc
        raw.columns=raw.columns.str.strip().str.upper()
        if raw.columns.duplicated().any():
            raise ValueError('Duplicate column names after header normalization. Give each source column a unique name.')
        required=['DATE','TIME','LOCATION','WEATHER','ROAD CONDITION','VEHICLE TYPE','CAUSE','SEVERITY','GENDER']
        missing=[c for c in required if c not in raw.columns]
        if missing:
            raise ValueError(f'Missing required columns: {missing}. Found: {list(raw.columns)}. '
                             'Upload a CSV with the documented RACC structure or adjust the column mapping.')
        if raw.empty:
            raise ValueError('The CSV contains a header but no records.')
        extra=[c for c in raw.columns if c not in required]
        if extra:
            print('Additional source columns are not analyzed by the current RACC modules:',extra)
        raw=raw[required].copy(deep=True)
        source_info=pd.DataFrame([{'filename':source_name,'bytes':len(source_bytes),'rows':len(raw),
            'columns':len(raw.columns),'sha256':SOURCE_SHA256,'encoding':CSV_ENCODING,
            'separator':CSV_SEPARATOR}])
        display(save_table(source_info,'00_uploaded_dataset_info'))
        print(f'Loaded uploaded dataset: {source_name} | rows: {len(raw)} | columns: {len(raw.columns)}')
        print('SHA-256:',SOURCE_SHA256)
        display(raw.head(6))
        return {'raw':raw,'required':required,'SOURCE_SHA256':SOURCE_SHA256,
                'source_name':source_name,'source_bytes':source_bytes}

    context = {
        'ALPHA': ALPHA,
        'APPLY_CONTEXT_DATE_FIXES': APPLY_CONTEXT_DATE_FIXES,
        'ASSUME_COMPLETE_REPORTING_CALENDAR': ASSUME_COMPLETE_REPORTING_CALENDAR,
        'COUNT_TEST_YEAR': COUNT_TEST_YEAR,
        'ColumnTransformer': ColumnTransformer,
        'ConfusionMatrixDisplay': ConfusionMatrixDisplay,
        'DROP_EXACT_DUPLICATES': DROP_EXACT_DUPLICATES,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'DummyClassifier': DummyClassifier,
        'DummyRegressor': DummyRegressor,
        'ExtraTreesClassifier': ExtraTreesClassifier,
        'FIRST_MONTH': FIRST_MONTH,
        'FORECAST_HORIZON': FORECAST_HORIZON,
        'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
        'INCLUDE_INVESTIGATED_CAUSE': INCLUDE_INVESTIGATED_CAUSE,
        'LAST_MONTH': LAST_MONTH,
        'LogisticRegression': LogisticRegression,
        'MIN_COMBINATION_RECORDS': MIN_COMBINATION_RECORDS,
        'MODULE_SPECS': MODULE_SPECS,
        'Markdown': Markdown,
        'N_BOOTSTRAP': N_BOOTSTRAP,
        'N_IMPORTANCE_REPEATS': N_IMPORTANCE_REPEATS,
        'N_PERMUTATIONS': N_PERMUTATIONS,
        'OUTPUT_ROOT': OUTPUT_ROOT,
        'OneHotEncoder': OneHotEncoder,
        'Path': Path,
        'Pipeline': Pipeline,
        'PoissonRegressor': PoissonRegressor,
        'PrecisionRecallDisplay': PrecisionRecallDisplay,
        'RandomForestClassifier': RandomForestClassifier,
        'RandomForestRegressor': RandomForestRegressor,
        'Ridge': Ridge,
        'RocCurveDisplay': RocCurveDisplay,
        'SEED': SEED,
        'SEVERITY_TEST_YEAR': SEVERITY_TEST_YEAR,
        'SimpleImputer': SimpleImputer,
        'StandardScaler': StandardScaler,
        'accuracy_score': accuracy_score,
        'binary_metrics': binary_metrics,
        'calibration_curve': calibration_curve,
        'classification_report': classification_report,
        'clone': clone,
        'count_metrics': count_metrics,
        'display': display,
        'export_text': export_text,
        'f1_score': f1_score,
        'itertools': itertools,
        'joblib': joblib,
        'json': json,
        'math': math,
        'mean_absolute_error': mean_absolute_error,
        'np': np,
        'optimize': optimize,
        'output_helpers': output_helpers,
        'pd': pd,
        'permutation_importance': permutation_importance,
        'plt': plt,
        'print': print,
        'roc_auc_score': roc_auc_score,
        'shutil': shutil,
        'sns': sns,
        'stats': stats,
        'versions': versions,
        'warnings': warnings,
    }

    runners={'00_setup':module_00_load_dataset}
    loaded_files=set()

    def load_task_code(module_id):
        if module_id in runners: return
        filename=MODULE_FILES[module_id]
        try:
            imported=importlib.import_module(filename)
        except ModuleNotFoundError as exc:
            if exc.name==filename:
                raise RuntimeError(f'Missing {filename}.py. Upload all four Python files into the same folder, '
                                   'then run racc_streamlit_app.py.') from exc
            raise
        runners.update(imported.build_modules(context))
        loaded_files.add(filename+'.py')


    def invalidate(module_id):
        invalid={module_id}
        changed=True
        while changed:
            changed=False
            for key,spec in MODULE_SPECS.items():
                if key not in invalid and invalid.intersection(spec['requires']):
                    invalid.add(key);changed=True
        for key in invalid:
            STATE.pop(key,None);EVENTS.pop(key,None);DURATIONS.pop(key,None)
            folder=OUTPUT_ROOT/MODULE_SPECS[key]['folder']
            if folder.is_dir(): shutil.rmtree(folder)

    def run_task(module_id):
        nonlocal ACTIVE_MODULE
        if module_id not in MODULE_SPECS: raise KeyError(f'Unknown module: {module_id}')
        missing=[k for k in MODULE_SPECS[module_id]['requires'] if k not in STATE]
        if missing: raise RuntimeError(f'Run prerequisite modules first: {missing}')
        with LOCK:
            load_task_code(module_id)
            invalidate(module_id)
            ACTIVE_MODULE=module_id
            EVENTS[module_id]=[]
            started=time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    result=runners[module_id](STATE)
                if not isinstance(result,dict): raise TypeError('A module must return named results.')
                STATE[module_id]=result
                for message in dict.fromkeys(str(w.message) for w in caught):
                    append_event('text','Warning: '+message)
                DURATIONS[module_id]=round(time.perf_counter()-started,3)
            finally:
                plt.close('all')
                ACTIVE_MODULE=None

    def ensure(module_id,force=False):
        for prerequisite in MODULE_SPECS[module_id]['requires']:
            ensure(prerequisite)
        if force or module_id not in STATE: run_task(module_id)

    def run_all():
        for key in MODULE_SPECS: ensure(key)

    def run_part(number):
        if number not in PART_TASKS: raise ValueError('Choose a part from 1 to 4.')
        for earlier in range(1,number):
            missing=[key for key in PART_TASKS[earlier] if key not in STATE]
            if missing:
                raise RuntimeError(f'Complete Part {earlier} before running Part {number}.')
        for key in PART_TASKS[number]: ensure(key)

    def part_ready(number):
        return all(key in STATE for earlier in range(1,number) for key in PART_TASKS[earlier])

    def status():
        return pd.DataFrame([{'module':key,'task':spec['title'],'completed':key in STATE,
            'seconds':DURATIONS.get(key),'prerequisites':', '.join(spec['requires']) or 'none'}
            for key,spec in MODULE_SPECS.items()])

    def archive_bytes():
        buffer=BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(OUTPUT_ROOT.rglob('*')):
                if p.is_file(): archive.write(p,p.relative_to(OUTPUT_ROOT))
        return buffer.getvalue()

    def cleanup(): owner.cleanup()

    return {'state':STATE,'events':EVENTS,'durations':DURATIONS,'root':OUTPUT_ROOT,
            'ensure':ensure,'run_task':run_task,'run_all':run_all,'run_part':run_part,
            'part_ready':part_ready,'loaded_files':loaded_files,'status':status,
            'archive_bytes':archive_bytes,'cleanup':cleanup,'source_name':source_filename,
            'source_sha256':hashlib.sha256(source_bytes).hexdigest(),'config':config}

PAGES={
    'Overview':None,'Dataset':'00_setup','Data cleaning':'01_cleaning',
    'Statistical analysis':'02_statistics','Exploratory analysis':'03_eda',
    'Classification':'04_classification','Numeric prediction':'05_prediction',
    'Forecasting':'06_forecasting','Model performance':'07_performance',
    'Features and combinations':'08_features','Downloads':'09_exports'}
TASK_NOTES={
    '00_setup':'Upload a CSV with the nine documented RACC columns. Additional columns are identified but not used.',
    '01_cleaning':'Review date corrections and category mappings. Repeated records are retained and flagged by default.',
    '02_statistics':'Frequencies, count summaries, year-blocked association tests, adjusted p-values and Wilson intervals.',
    '03_eda':'Category distributions, missingness, severity vocabulary, time patterns and predictor associations.',
    '04_classification':'Binary and four-class injury-grade models. Train on 2021–2023; reserve 2024 for evaluation.',
    '05_prediction':'Predict daily recorded-row counts one day ahead using past-only lags and calendar variables. Test on 2025.',
    '06_forecasting':'Predict monthly recorded-row counts from one fixed origin, then extrapolate beyond the dataset end.',
    '07_performance':'Reserved-year results, baseline comparisons, calibration, uncertainty, residuals and cleaning sensitivity.',
    '08_features':'Variable importance, feature-set ablations, supported combinations, reference rules and illustrative predictions.',
    '09_exports':'Download the generated tables, figures, forecasts, settings and fitted models, organized by task.'}

def clear_engine():
    previous=st.session_state.pop('racc_engine',None)
    if previous is not None: previous['cleanup']()
    st.session_state.pop('racc_fingerprint',None)
    st.session_state.pop('racc_error',None)

def show_failure(exc):
    st.session_state['racc_error']={'message':str(exc),'details':traceback.format_exc()}

def run_action(engine,key=None,force=False):
    st.session_state.pop('racc_error',None)
    try:
        label='Running the complete analysis…' if key is None else 'Running this task and any missing prerequisites…'
        with st.spinner(label):
            if key is None: engine['run_all']()
            else: engine['ensure'](key,force=force)
        st.rerun()
    except Exception as exc:
        show_failure(exc)

def nice_title(value):
    if not value: return ''
    return __import__('re').sub(r'^\d+[a-z]?_','',value).replace('_',' ').capitalize()

def show_events(engine,module_id):
    texts=[]
    for i,event in enumerate(engine['events'].get(module_id,[])):
        title=nice_title(event['title'])
        if event['kind']=='table':
            if title: st.subheader(title)
            table=event['data']
            st.dataframe(table,width='stretch',hide_index=True)
            st.download_button('Download this table',table.to_csv(index=False).encode('utf-8'),
                file_name=(event['title'] or f'{module_id}_table_{i}')+'.csv',mime='text/csv',
                key=f'table_{module_id}_{i}',on_click='ignore')
        elif event['kind']=='image':
            st.image(event['data'],caption=title,width='stretch')
        elif event['kind']=='markdown': st.markdown(event['data'])
        else: texts.append(event['data'])
    if texts:
        with st.expander('Execution notes and detailed rules'):
            st.code('\n'.join(texts),language=None)

def prediction_form(engine):
    if '08_features' not in engine['state']: return
    st.divider();st.subheader('Try a severity prediction')
    st.caption('An illustration among reported incidents. Probabilities may be poorly calibrated; review the performance module.')
    cleaned=engine['state']['01_cleaning']['df']
    raw=engine['state']['00_setup']['raw']
    with st.form('severity_prediction_form'):
        c1,c2=st.columns(2)
        with c1:
            date=st.date_input('Incident date',value=__import__('datetime').date(2024,8,15))
            clock=st.time_input('Incident time',value=__import__('datetime').time(20,0))
            location=st.text_input('Location',value='P1 San Teodoro')
        with c2:
            weather=st.selectbox('Weather',sorted(cleaned.weather.dropna().unique()))
            road=st.selectbox('Road condition',sorted(raw['ROAD CONDITION'].dropna().str.strip().unique()))
            vehicle=st.text_input('Vehicle type',value='Motorcycle')
            gender=st.selectbox('Recorded gender composition',['Male','Female','Male & Female','Male & Male','Unknown'])
        cause=st.text_input('Investigated cause',value='Collision') if engine['config']['include_investigated_cause'] else None
        submitted=st.form_submit_button('Predict recorded injury severity')
    if submitted:
        record={'DATE':date.strftime('%d/%m/%Y'),'TIME':clock.strftime('%I:%M %p'),
                'LOCATION':location,'WEATHER':weather,'ROAD CONDITION':road,'VEHICLE TYPE':vehicle,'GENDER':gender}
        if cause is not None: record['CAUSE']=cause
        try:
            result=engine['state']['08_features']['predict_record_severity'](record)
            st.metric('Model probability: major injury or fatal',f"{result['probability_major_or_fatal']:.1%}")
            st.write('Predicted recorded class:', 'major injury / fatal' if result['predicted_class'] else 'minor / no injury')
        except Exception as exc: st.error(str(exc))

def show_overview(engine):
    if engine is None:
        st.info('Upload your RACC CSV in the left panel, then click Load dataset.')
        st.markdown('**Workflow:** upload → review cleaning → run analyses → inspect performance → download results.')
        st.markdown('The app includes statistical analysis, EDA, classification, numeric prediction, forecasting, performance evaluation and feature combinations.')
        with st.expander('Required CSV format',expanded=True):
            st.code(','.join(REQUIRED_COLUMNS),language='text')
            st.write('Designed for the original 2021–2025 RACC data. Another schema or recording period requires changes to the date rules and evaluation windows.')
        return
    state=engine['state'];raw=state['00_setup']['raw']
    a,b,c=st.columns(3)
    a.metric('Uploaded records',f'{len(raw):,}')
    b.metric('Variables',len(raw.columns))
    c.metric('Completed modules',f'{len(state)}/10')
    st.caption(f"Dataset: {engine['source_name']}")
    if '01_cleaning' in state:
        d=state['01_cleaning']['df']
        st.write(f"Usable injury grades: {int(d.severity_grade.notna().sum()):,}. Context-dependent date corrections: {int(d.provisional_date_flag.sum())}.")
    st.dataframe(engine['status'](),width='stretch',hide_index=True)
    if '07_performance' in state:
        st.subheader('Reserved-year performance')
        st.dataframe(state['07_performance']['performance_dashboard'],width='stretch',hide_index=True)
    if '09_exports' in state:
        st.markdown(state['09_exports']['run_summary'])
    else:
        st.info('Select a task on the left, or run the complete analysis. Completed results remain available while you navigate.')

def show_downloads(engine):
    state=engine['state']
    if '09_exports' not in state:
        st.info('Generate the consolidated exports to include all task results and fitted models. You can also download completed outputs below.')
    if st.button('Generate or refresh complete exports',key='generate_exports',type='primary'):
        run_action(engine,'09_exports',force=True)
    files=[p for p in sorted(engine['root'].rglob('*')) if p.is_file()]
    if not files: return
    st.download_button('Download all available results (ZIP)',engine['archive_bytes'](),
        file_name='RACC_analysis_results.zip',mime='application/zip',key='all_outputs_zip',on_click='ignore')
    names=[str(p.relative_to(engine['root'])) for p in files]
    chosen=st.selectbox('Download an individual output',names,key='individual_output')
    selected=engine['root']/chosen
    mime={'.csv':'text/csv','.png':'image/png','.json':'application/json','.md':'text/markdown','.txt':'text/plain'}.get(selected.suffix,'application/octet-stream')
    st.download_button('Download selected file',selected.read_bytes(),file_name=selected.name,mime=mime,
                       key='individual_output_download',on_click='ignore')
    st.caption('Joblib models should be loaded only from trusted sources in a compatible Python environment. Uploaded data and results are held in a private temporary session directory.')



PART_LABELS={
    'Part 1 — Upload and setup':1,
    'Part 2 — Data analysis':2,
    'Part 3 — Models and forecasting':3,
    'Part 4 — Evaluation and downloads':4}
PART_PAGES={
    1:{'Dataset':'00_setup'},
    2:{'Data cleaning':'01_cleaning','Statistical analysis':'02_statistics','Exploratory analysis':'03_eda'},
    3:{'Classification':'04_classification','Numeric prediction':'05_prediction','Forecasting':'06_forecasting'},
    4:{'Model performance':'07_performance','Features and combinations':'08_features','Downloads':'09_exports'}}

def run_part_action(engine,number):
    st.session_state.pop('racc_error',None)
    try:
        with st.spinner(f'Running Part {number}…'):
            engine['run_part'](number)
        st.rerun()
    except Exception as exc: show_failure(exc)

def main():
    st.set_page_config(page_title='RACC — Four-part analysis',page_icon='📊',layout='wide')
    st.title('Road-accident analysis')
    st.caption('Four parts · Upload once, then run each analysis stage separately.')
    st.sidebar.header('Dataset')
    uploaded=st.sidebar.file_uploader('Upload RACC dataset',type=['csv'],accept_multiple_files=False,key='racc_upload')
    config=dict(DEFAULT_CONFIG)
    with st.sidebar.expander('Analysis settings'):
        config['apply_context_date_fixes']=st.checkbox('Apply documented provisional date fixes',True,key='fix_dates')
        config['drop_exact_duplicates']=st.checkbox('Remove exact repeated rows',False,key='drop_duplicates')
        config['include_investigated_cause']=st.checkbox('Use investigated cause as a predictor',False,key='include_cause')
        config['assume_complete_reporting_calendar']=st.checkbox('Assume complete reporting calendar',True,key='complete_calendar')
        st.caption('Unlisted dates count as zero reported rows under this assumption. Repeated rows may represent different people or vehicles.')
        config['forecast_horizon']=st.slider('Forecast months after dataset end',1,24,12,key='forecast_months')
        config['n_permutations']=int(st.number_input('Association-test permutations',min_value=199,max_value=4999,value=999,step=200,key='permutations'))
        config['n_bootstrap']=int(st.number_input('Bootstrap samples',min_value=100,max_value=2000,value=500,step=100,key='bootstrap'))
        config['n_importance_repeats']=int(st.number_input('Importance repeats',min_value=2,max_value=30,value=10,step=1,key='importance_repeats'))
        config['csv_encoding']=st.selectbox('CSV encoding',['utf-8-sig','utf-8','cp1252','latin-1'],key='csv_encoding')
        separator=st.selectbox('CSV separator',['Comma','Semicolon','Tab'],key='csv_separator')
        config['csv_separator']={'Comma':',','Semicolon':';','Tab':'\t'}[separator]
    fingerprint=None
    if uploaded is not None:
        fingerprint=hashlib.sha256(uploaded.getvalue()+uploaded.name.encode('utf-8')+
                                   json.dumps(config,sort_keys=True).encode()).hexdigest()
    if st.session_state.get('racc_fingerprint')!=fingerprint: clear_engine()
    engine=st.session_state.get('racc_engine')
    if st.sidebar.button('Load dataset — Part 1',disabled=uploaded is None,key='load_dataset',type='primary'):
        candidate=None
        try:
            candidate=create_engine(uploaded.getvalue(),uploaded.name,config,plot_lock=shared_plot_lock())
            candidate['ensure']('00_setup')
            if engine is not None: engine['cleanup']()
            st.session_state['racc_engine']=candidate
            st.session_state['racc_fingerprint']=fingerprint
            st.session_state.pop('racc_error',None)
            st.rerun()
        except Exception as exc:
            if candidate is not None: candidate['cleanup']()
            show_failure(exc)
    engine=st.session_state.get('racc_engine')
    selected=st.sidebar.radio('Choose a part',list(PART_LABELS),key='part_navigation')
    number=PART_LABELS[selected]
    if engine is not None:
        st.sidebar.progress(len(engine['state'])/10,text=f"{len(engine['state'])} of 10 tasks complete")
    if 'racc_error' in st.session_state:
        failure=st.session_state['racc_error'];st.error(failure['message'])
        with st.expander('Error details'):st.code(failure['details'])
    if engine is None:
        st.info('Start with Part 1: upload your CSV in the left panel, then click Load dataset — Part 1.')
        with st.expander('Required CSV columns',expanded=True):
            st.code(','.join(REQUIRED_COLUMNS),language='text')
        st.markdown('**Part 1:** upload and validate. **Part 2:** clean and explore. **Part 3:** fit models. **Part 4:** evaluate and download.')
        st.caption('All four Python files must be in the same project folder. Launch racc_streamlit_app.py.')
        return
    if number==1:
        st.header('Part 1 — Upload and setup')
        show_overview(engine)
        with st.expander('Uploaded dataset details'):show_events(engine,'00_setup')
        st.success('Dataset loaded. Select Part 2 to clean and analyze it.')
    else:
        st.header(selected)
        ready=engine['part_ready'](number)
        complete=all(key in engine['state'] for key in PART_TASKS[number])
        if not ready:st.info(f'Complete Part {number-1} before starting Part {number}.')
        label=f'Part {number} completed — resume tasks' if complete else f'Run Part {number}'
        if st.button(label,key=f'run_part_{number}',disabled=not ready,type='primary'):
            run_part_action(engine,number)
        choices=PART_PAGES[number]
        task=st.selectbox('Task results',list(choices),key=f'task_part_{number}')
        key=choices[task]
        st.write(TASK_NOTES[key])
        if key=='09_exports':
            if ready:show_downloads(engine)
        elif key in engine['state']:
            show_events(engine,key)
            if key=='08_features':prediction_form(engine)
        else:
            st.info(f'Run Part {number} to display this task’s results.')
        if complete and number<4:st.success(f'Part {number} is complete. You can continue to Part {number+1}.')
    st.divider()
    st.caption('Reported rows are not verified unique accidents or exposure-adjusted rates. Review date repairs, reporting completeness and weak predictive results. Defaults target January 2021–December 2025.')

if __name__=='__main__':
    main()
