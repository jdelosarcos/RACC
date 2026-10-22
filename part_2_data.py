"""Part 2/4: cleaning, statistical analysis and exploratory data analysis.

Keep this file beside racc_streamlit_app.py. The main app imports it only when needed.
Analysis settings and output writers are bound to the current browser session.
"""
from __future__ import annotations

def build_modules(context):
    """Bind shared settings/services and return the task functions in this part."""
    ALPHA = context['ALPHA']
    APPLY_CONTEXT_DATE_FIXES = context['APPLY_CONTEXT_DATE_FIXES']
    ASSUME_COMPLETE_REPORTING_CALENDAR = context['ASSUME_COMPLETE_REPORTING_CALENDAR']
    DROP_EXACT_DUPLICATES = context['DROP_EXACT_DUPLICATES']
    FIRST_MONTH = context['FIRST_MONTH']
    LAST_MONTH = context['LAST_MONTH']
    N_PERMUTATIONS = context['N_PERMUTATIONS']
    OUTPUT_ROOT = context['OUTPUT_ROOT']
    SEED = context['SEED']
    SEVERITY_TEST_YEAR = context['SEVERITY_TEST_YEAR']
    display = context['display']
    itertools = context['itertools']
    np = context['np']
    output_helpers = context['output_helpers']
    pd = context['pd']
    plt = context['plt']
    print = context['print']
    sns = context['sns']
    stats = context['stats']

    def module_01_clean_data(state):
        """Audit data quality, normalize categories, review date repairs, retain duplicate flags and build analysis-ready count series.

        Inputs: Raw CSV rows from Module 0.
        Returns: a dictionary of named module outputs."""
        OUTPUT_DIR = OUTPUT_ROOT / '01_data_cleaning'
        save_table, finish_figure = output_helpers(OUTPUT_DIR)
        raw = state['00_setup']['raw']
        required = state['00_setup']['required']

        raw_audit = pd.DataFrame({
            'dtype':raw.dtypes.astype(str), 'missing_n':raw.isna().sum(),
            'missing_pct':100*raw.isna().mean(), 'unique_nonmissing':raw.nunique(),
            'whitespace_affected_n':[int((raw[c].notna() & raw[c].ne(raw[c].str.strip())).sum()) for c in raw]
        }).rename_axis('variable').reset_index()
        display(save_table(raw_audit, '01_raw_data_audit'))
        print('Exact repeated rows after their first occurrence:', int(raw.duplicated().sum()))
        display(raw.loc[raw.duplicated(keep=False)].head(14))
        raw_counts = pd.concat([raw[c].value_counts(dropna=False).rename('n').reset_index()
            .rename(columns={c:'value'}).assign(variable=c) for c in raw], ignore_index=True)
        save_table(raw_counts[['variable','value','n']], '02_raw_category_frequencies')

        # Explicit reviewable corrections, not a general automatic year repair.
        DATE_FIXES = {
            '8/18/0201': ('2021-08-18','Between 15/08/2021 and 23/08/2021; US-style date and year typo'),
            '1/4/202':   ('2022-01-04','Between 01/01/2022 and 10/01/2022; truncated year and US-style date'),
            '1/14/202':  ('2022-01-14','January 2022 neighbors; truncated year and unambiguous US-style date'),
            '1/16/202':  ('2022-01-16','January 2022 neighbors; truncated year and unambiguous US-style date'),
            '11/2/1014': ('2024-11-02','Between 28/10/2024 and 02/11/2024; assumed month/day and year typo'),
            '8/3/1025':  ('2025-08-03','Between 01/08/2025 and later August records; assumed US-style date/year typo'),
            '8/10/1025': ('2025-08-10','Before 14/08/2025; assumed US-style date/year typo'),
        }

        def normalize_text(s):
            s = s.astype('string').str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
            return s.replace({'':pd.NA, 'n/a':pd.NA, 'na':pd.NA, 'null':pd.NA, 'none':pd.NA})

        def parse_date(value):
            if pd.isna(value):
                return pd.NaT, 'missing', None
            value = str(value).strip()
            if value in DATE_FIXES:
                proposed, reason = DATE_FIXES[value]
                return (pd.Timestamp(proposed) if APPLY_CONTEXT_DATE_FIXES else pd.NaT,
                        'provisional_context_fix' if APPLY_CONTEXT_DATE_FIXES else 'quarantined_context_fix', proposed)
            compact = __import__('re').sub(r'/+', '/', value)
            parts = compact.split('/')
            try:
                a, b, y = map(int, parts)
                if not (2021 <= y <= 2025):
                    return pd.NaT, 'invalid_or_out_of_scope_year', None
                month_first = b > 12 and a <= 12
                date = pd.Timestamp(year=y, month=a if month_first else b, day=b if month_first else a)
                status = 'unambiguous_month_first' if month_first else 'dominant_day_first'
                if compact != value:
                    status += '_repeated_slash_removed'
                return date, status, None
            except (ValueError, TypeError):
                return pd.NaT, 'invalid_date', None

        def parse_hour(value):
            if pd.isna(value): return np.nan
            for fmt in ['%I:%M %p','%I:%M:%S %p','%I:%M%p','%H:%M','%H:%M:%S']:
                try:
                    t = __import__('datetime').datetime.strptime(str(value).strip().upper(), fmt)
                    return t.hour + t.minute/60 + t.second/3600
                except ValueError:
                    continue
            return np.nan

        def clean_location(s):
            s = normalize_text(s).str.replace(r'\bp\s*-?\s*(?=\d)', 'p', regex=True)
            aliases = {'bunawna':'bunawan','san trodoro':'san teodoro',
                       'pobalcion':'poblacion','mamabalili':'mambalili'}
            for a,b in aliases.items(): s = s.str.replace(a,b,regex=False)
            return s

        def area_from_location(v):
            if pd.isna(v): return 'unknown'
            # Match named areas explicitly; do not infer a barangay from a sitio alone.
            for area in ['san teodoro','san andres','san marcos','bunawan brook',
                         'consuelo','libertad','poblacion','mambalili','imelda','nueva era']:
                if area in v: return area
            if any(t in v for t in ['bun. brook','bun, brook','b-brook']): return 'bunawan brook'
            if __import__('re').search(r'\bpob\.?\b',v): return 'poblacion'
            return 'other / unresolved'

        def vehicle_group(v):
            if pd.isna(v): return 'unknown'
            if __import__('re').search(r'\bvs\.?|&|\band\b|^2\b',v): return 'multiple vehicles recorded'
            if v in ['motorcycle','single motor','motor single','motorcyle','motorcycle aid']: return 'motorcycle'
            if 'pedestrian' in v: return 'pedestrian'
            if v in ['tricycle','side car','sidecar']: return 'tricycle / sidecar'
            if v in ['truck','wing van','wingvan','bus']: return 'truck / bus'
            if v in ['car','suv','van','mini van','wagon','pick up','multicab','l300'] or '4 wheeled' in v:
                return 'car / utility / van'
            if v in ['vehicle','vehicular']: return 'vehicle unspecified'
            return 'other / ambiguous'

        def cause_group(v):
            if pd.isna(v): return 'unknown'
            if any(t in v for t in ['alcohol','aid']): return 'alcohol mentioned / abbreviation (unverified)'
            if any(t in v for t in ['dog','animal']): return 'animal related'
            if 'pedestrian' in v: return 'pedestrian related'
            if any(t in v for t in ['tire','break','brake']): return 'mechanical / braking mentioned'
            if any(t in v for t in ['road condition','weather','visibility']): return 'road / weather / visibility'
            if any(t in v for t in ['reckless','overspeed','human error','asleep']): return 'driving behavior mentioned'
            if any(t in v for t in ['self','sels','overshoot']): return 'self accident / overshoot'
            if any(t in v for t in ['collision','swip','bump','hit','bangga','vs.','vs ']): return 'collision / contact'
            return 'other / unspecified'

        def gender_composition(v):
            if pd.isna(v): return 'unknown'
            tokens = __import__('re').findall(r'\b(?:female|male)\b',v)
            sexes = set(tokens)
            if 'others' in v: return 'mixed + unspecified' if len(sexes)==2 else 'partly unspecified'
            if sexes=={'male','female'}: return 'mixed'
            if sexes=={'male'}: return 'male only'
            if sexes=={'female'}: return 'female only'
            return 'unknown'

        def clean_records(frame):
            d = frame.copy(deep=True).reset_index(drop=True)
            for c in required:
                if c not in d: d[c] = pd.NA
            d['source_row'] = np.arange(len(d)) + 2
            d['exact_duplicate_flag'] = d[required].duplicated(keep=False)
            for c in required:
                d[c.lower().replace(' ','_')+'_raw'] = d[c]
            for c in required: d[c] = normalize_text(d[c])
            dates = [parse_date(v) for v in d['date_raw']]
            d['date'] = pd.to_datetime([v[0] for v in dates])
            d['date_status'] = [v[1] for v in dates]
            d['proposed_date'] = [v[2] for v in dates]
            d['provisional_date_flag'] = d['date_raw'].isin(DATE_FIXES)
            d['hour'] = d['TIME'].map(parse_hour)
            d['invalid_time_flag'] = d['TIME'].notna() & d['hour'].isna()
            d['location_clean'] = clean_location(d['LOCATION'])
            d['area'] = d['location_clean'].map(area_from_location)
            d['weather'] = d['WEATHER']
            d['road_condition'] = d['ROAD CONDITION'].replace({'clear':'clear (unverified)'})
            d['vehicle_clean'] = d['VEHICLE TYPE'].str.replace(r'pick\s+up','pick up',regex=True)
            d['vehicle_group'] = d['vehicle_clean'].map(vehicle_group)
            d['cause_group'] = d['CAUSE'].map(cause_group)
            d['gender_composition'] = d['GENDER'].map(gender_composition)
            d['severity_clean'] = d['SEVERITY'].replace({'minor injry':'minor injury'})
            injury_labels = ['no injury','minor injury','major injury','fatal']
            d['severity_grade'] = d['severity_clean'].where(d['severity_clean'].isin(injury_labels))
            d['severe'] = d['severity_grade'].map({'no injury':0,'minor injury':0,'major injury':1,'fatal':1})
            d['year'] = d['date'].dt.year
            d['month'] = d['date'].dt.month
            d['day_of_week'] = d['date'].dt.dayofweek
            d['is_weekend'] = d['day_of_week'].isin([5,6]).astype(float).where(d['date'].notna())
            for col,period in [('hour',24),('month',12),('day_of_week',7)]:
                d[col+'_sin'] = np.sin(2*np.pi*d[col]/period)
                d[col+'_cos'] = np.cos(2*np.pi*d[col]/period)
            # Provisional key: same date + exact parsed time + normalized location. Never use severity.
            complete = d['date'].notna() & d['hour'].notna() & d['location_clean'].notna()
            d['incident_proxy'] = ('row_'+d['source_row'].astype(str))
            d.loc[complete,'incident_proxy'] = (d.loc[complete,'date'].dt.strftime('%Y-%m-%d')+'|'+
                d.loc[complete,'hour'].round(6).astype(str)+'|'+d.loc[complete,'location_clean'])
            if DROP_EXACT_DUPLICATES:
                d = d.loc[~frame[required].reset_index(drop=True).duplicated()].copy()
            return d

        df = clean_records(raw)
        date_review = df.loc[~df['date_status'].eq('dominant_day_first'),
            ['source_row','date_raw','date','date_status','proposed_date','location_raw']].copy()
        date_review['reason'] = date_review['date_raw'].map({k:v[1] for k,v in DATE_FIXES.items()})
        display(save_table(date_review,'03_date_review'))
        print('Provisional context-dependent dates:', int(df.provisional_date_flag.sum()),
              '| Unusable dates:', int(df.date.isna().sum()))
        display(df.groupby('date_status',dropna=False).size().rename('records').to_frame())

        mapping_tables = []
        for original,cleaned in [('LOCATION','location_clean'),('ROAD CONDITION','road_condition'),
            ('VEHICLE TYPE','vehicle_group'),('CAUSE','cause_group'),('GENDER','gender_composition'),('SEVERITY','severity_clean')]:
            t = pd.DataFrame({'original':raw[original].astype(object),'cleaned':df[cleaned].reindex(raw.index).astype(object)})
            mapping_tables.append(t.groupby(['original','cleaned'],dropna=False).size().rename('n')
                                  .reset_index().assign(variable=original))
        cleaning_mappings = pd.concat(mapping_tables,ignore_index=True)
        save_table(cleaning_mappings,'04_cleaning_mapping_audit')
        display(cleaning_mappings.loc[cleaning_mappings.variable.eq('SEVERITY')])
        print('Unparseable nonmissing times:', int(df.invalid_time_flag.sum()))
        print('Records with no usable injury grade:', int(df.severity_grade.isna().sum()))
        print('Record count:',len(df),'| distinct provisional incident keys:',df.incident_proxy.nunique())
        save_table(df,'05_cleaned_records_with_raw_values')

        eligibility = pd.DataFrame({
            'condition':['all records','usable date','usable injury grade','severity development (2021–2023)',
                         'severity holdout (2024)','2025 status/schema quarantine'],
            'n':[len(df),df.date.notna().sum(),df.severity_grade.notna().sum(),
                 (df.severity_grade.notna() & df.year.lt(SEVERITY_TEST_YEAR)).sum(),
                 (df.severity_grade.notna() & df.year.eq(SEVERITY_TEST_YEAR)).sum(),df.year.ge(2025).sum()]
        })
        display(save_table(eligibility,'06_model_eligibility'))

        injury_era = df.loc[df.severity_grade.notna() & df.year.between(2021,2024)].copy()
        association_features=['weather','road_condition','vehicle_group','area','gender_composition','cause_group']
        def consistent_or_mixed(s):
            values=s.fillna('missing').unique()
            return values[0] if len(values)==1 else 'mixed across records'
        agg={'severe':'max','year':'first',**{c:consistent_or_mixed for c in association_features}}
        stat_df=injury_era.groupby('incident_proxy',sort=False).agg(agg).reset_index()

        if not ASSUME_COMPLETE_REPORTING_CALENDAR:
            raise ValueError('Confirm coverage or provide a reporting-coverage calendar before count modeling.')
        calendar = pd.date_range(pd.Timestamp(FIRST_MONTH), pd.Timestamp(LAST_MONTH)+pd.offsets.MonthEnd(0),freq='D')
        dated = df.loc[df.date.notna()].copy()
        outside_calendar = dated.loc[~dated.date.between(calendar.min(),calendar.max())]
        assert outside_calendar.empty, 'Update FIRST_MONTH/LAST_MONTH: dates fall outside configured calendar.'
        daily = pd.DataFrame(index=calendar)
        daily.index.name = 'date'
        daily['record_count'] = dated.groupby('date').size().reindex(calendar,fill_value=0).astype(int)
        daily['proxy_count'] = dated.groupby('date').incident_proxy.nunique().reindex(calendar,fill_value=0).astype(int)
        strict_dates = dated.loc[~dated.provisional_date_flag]
        daily['strict_date_record_count'] = strict_dates.groupby('date').size().reindex(calendar,fill_value=0).astype(int)
        monthly = daily.resample('MS').sum()
        save_table(daily.reset_index(),'09_daily_record_counts')
        save_table(monthly.reset_index(),'10_monthly_count_sensitivity')

        exported_names = ['df', 'DATE_FIXES', 'clean_records', 'raw_audit', 'raw_counts', 'date_review', 'cleaning_mappings', 'eligibility', 'calendar', 'daily', 'monthly', 'injury_era', 'stat_df', 'association_features']
        return {name:value for name,value in locals().items() if name in exported_names}

    def module_02_statistical_analysis(state):
        """Calculate descriptive statistics, categorical associations, adjusted p-values and severity-rate intervals.

        Inputs: Cleaned records, daily counts and provisional incident groups from Module 1.
        Returns: a dictionary of named module outputs."""
        OUTPUT_DIR = OUTPUT_ROOT / '02_statistical_analysis'
        save_table, finish_figure = output_helpers(OUTPUT_DIR)
        association_features = state['01_cleaning']['association_features']
        daily = state['01_cleaning']['daily']
        df = state['01_cleaning']['df']
        stat_df = state['01_cleaning']['stat_df']

        analysis_categories = ['weather','road_condition','vehicle_group','area','gender_composition','severity_clean']
        frequency_tables = []
        for c in analysis_categories:
            t = df[c].fillna('missing').value_counts().rename('n').to_frame()
            t['percent'] = 100*t.n/len(df)
            frequency_tables.append(t.rename_axis('value').reset_index().assign(variable=c))
        frequencies = pd.concat(frequency_tables,ignore_index=True)
        save_table(frequencies[['variable','value','n','percent']],'07_cleaned_frequencies')
        display(frequencies.loc[frequencies.variable.eq('severity_clean')])

        display(daily.describe().T)
        count_summary = pd.DataFrame({'mean':daily.mean(),'median':daily.median(),
            'variance':daily.var(),'zero_day_fraction':daily.eq(0).mean(),'max':daily.max()})
        count_summary['variance_to_mean'] = count_summary.variance/count_summary['mean']
        display(save_table(count_summary.reset_index(names='count_definition'),'11_daily_count_summary'))

        def wilson_interval(k,n,alpha=0.05):
            if n==0: return np.nan,np.nan
            z=stats.norm.ppf(1-alpha/2);p=k/n;den=1+z*z/n
            mid=(p+z*z/(2*n))/den
            half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
            return max(0,mid-half),min(1,mid+half)

        def bh_adjust(pvalues):
            p=np.asarray(pvalues,float);order=np.argsort(p);m=len(p)
            adjusted=np.empty(m);adjusted[order]=np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1]
            return np.minimum(adjusted,1)

        test_rows=[]
        for c in association_features:
            values=stat_df[c].fillna('missing');freq=values.value_counts()
            values=values.where(values.map(freq).ge(10),'other (rare pooled)')
            x,levels=pd.factorize(values);y=stat_df.severe.to_numpy(dtype=int);r=len(levels)
            observed=np.bincount(x*2+y,minlength=r*2).reshape(r,2)
            chi,p_asym,dof,expected=stats.chi2_contingency(observed,correction=False)
            def chi_stat(labels):
                counts=np.bincount(x*2+labels,minlength=r*2).reshape(r,2)
                return float(np.sum((counts-expected)**2/expected))
            permutation_stats=[]
            local_rng=np.random.default_rng(SEED)
            year_indices=[np.flatnonzero(stat_df.year.to_numpy()==yr) for yr in sorted(stat_df.year.unique())]
            for _ in range(N_PERMUTATIONS):
                perm=y.copy()
                for idx in year_indices: perm[idx]=local_rng.permutation(y[idx])
                permutation_stats.append(chi_stat(perm))
            perm_p=(1+np.sum(np.asarray(permutation_stats)>=chi-1e-12))/(N_PERMUTATIONS+1)
            test_rows.append({'feature':c,'proxy_incidents':len(y),'levels_after_pooling':r,
                'chi_square':chi,'cramers_v':np.sqrt(chi/(len(y)*min(r-1,1))),
                'min_expected_count':expected.min(),'expected_below_5_pct':100*np.mean(expected<5),
                'p_chi_square_asymptotic':p_asym,'p_permutation_within_year':perm_p})
        association_tests=pd.DataFrame(test_rows)
        association_tests['p_BH_adjusted']=bh_adjust(association_tests.p_permutation_within_year)
        association_tests['passes_FDR_0_05']=association_tests.p_BH_adjusted.lt(ALPHA)
        display(save_table(association_tests.sort_values('p_BH_adjusted'),'12_exploratory_association_tests'))

        area_rates=stat_df.groupby('area').severe.agg(['size','sum','mean']).rename(columns={'size':'proxy_n','sum':'severe_n','mean':'severe_fraction'})
        area_rates[['wilson_low','wilson_high']]=pd.DataFrame([wilson_interval(r.severe_n,r.proxy_n) for r in area_rates.itertuples()],index=area_rates.index)
        display(save_table(area_rates.reset_index().sort_values('proxy_n',ascending=False),'13_area_severity_intervals'))

        exported_names = ['frequencies', 'count_summary', 'association_tests', 'area_rates']
        return {name:value for name,value in locals().items() if name in exported_names}

    def module_03_exploratory_analysis(state):
        """Visualize category distributions, missingness, changes in labels, time patterns and associations.

        Inputs: Cleaned records and daily/monthly count series from Module 1.
        Returns: a dictionary of named module outputs."""
        OUTPUT_DIR = OUTPUT_ROOT / '03_exploratory_analysis'
        save_table, finish_figure = output_helpers(OUTPUT_DIR)
        association_features = state['01_cleaning']['association_features']
        df = state['01_cleaning']['df']
        monthly = state['01_cleaning']['monthly']
        stat_df = state['01_cleaning']['stat_df']

        analysis_categories = ['weather','road_condition','vehicle_group','area','gender_composition','severity_clean']

        fig,axes = plt.subplots(2,3,figsize=(16,10))
        for ax,c in zip(axes.flat,analysis_categories):
            counts = df[c].fillna('missing').value_counts().head(10).sort_values()
            counts.plot.barh(ax=ax,color='#147d92')
            ax.set(title=c.replace('_',' ').title(),xlabel='Reported records',ylabel='')
        finish_figure('01_category_distributions')

        severity_by_year = pd.crosstab(df.year,df.severity_clean.fillna('missing'))
        display(save_table(severity_by_year.reset_index(),'08_severity_vocabulary_by_year'))
        severity_by_year.div(severity_by_year.sum(axis=1),axis=0).plot.bar(stacked=True,figsize=(12,5))
        plt.ylabel('Proportion of records');plt.title('Severity vocabulary changes: consciousness is not an injury grade')
        plt.legend(bbox_to_anchor=(1.02,1),loc='upper left')
        finish_figure('02_severity_schema_shift')

        missing = df[['date','hour','location_clean','weather','road_condition','vehicle_clean','severity_grade']].isna()
        missing.mean().mul(100).sort_values().plot.barh(color='#ad5b28')
        plt.xlabel('Missing / unusable (%)');plt.title('Missingness after deterministic cleaning')
        finish_figure('03_missingness')

        hour_day = pd.crosstab(df.day_of_week, np.floor(df.hour))
        hour_day = hour_day.reindex(index=range(7),columns=range(24),fill_value=0)
        plt.figure(figsize=(14,4));sns.heatmap(hour_day,cmap='YlOrRd',yticklabels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
        plt.xlabel('Hour of day');plt.ylabel('Day');plt.title('Reported records by weekday and hour; missing times omitted')
        finish_figure('04_weekday_hour_heatmap')

        monthly[['record_count','proxy_count','strict_date_record_count']].plot(marker='o',markersize=3,figsize=(13,5))
        plt.title('Monthly recording volume and count-definition sensitivity');plt.ylabel('Records / provisional keys')
        finish_figure('05_monthly_record_counts')
        month_matrix = monthly.assign(year=monthly.index.year,month=monthly.index.month).pivot(index='year',columns='month',values='record_count')
        sns.heatmap(month_matrix,annot=True,fmt='g',cmap='Blues');plt.title('Monthly reported rows by year')
        finish_figure('06_year_month_heatmap')

        association_matrix=pd.DataFrame(index=association_features,columns=association_features,dtype=float)
        for a,b in itertools.product(association_features,repeat=2):
            table=pd.crosstab(stat_df[a],stat_df[b]);n=table.to_numpy().sum()
            if a==b: value=1.0
            elif min(table.shape)>1:
                chi=stats.chi2_contingency(table,correction=False)[0]
                phi2=max(0,chi/n-((table.shape[0]-1)*(table.shape[1]-1))/(n-1))
                rr=table.shape[0]-(table.shape[0]-1)**2/(n-1)
                cc=table.shape[1]-(table.shape[1]-1)**2/(n-1)
                value=np.sqrt(phi2/min(rr-1,cc-1)) if min(rr-1,cc-1)>0 else np.nan
            else: value=np.nan
            association_matrix.loc[a,b]=value
        save_table(association_matrix.reset_index(names='feature'),'14_predictor_association_matrix')
        sns.heatmap(association_matrix,annot=True,vmin=0,vmax=1,cmap='Blues',fmt='.2f')
        plt.title('Bias-corrected categorical association (Cramér’s V)')
        finish_figure('07_predictor_associations')

        exported_names = ['severity_by_year', 'hour_day', 'association_matrix', 'month_matrix']
        return {name:value for name,value in locals().items() if name in exported_names}

    return {'01_cleaning': module_01_clean_data, '02_statistics': module_02_statistical_analysis, '03_eda': module_03_exploratory_analysis}
