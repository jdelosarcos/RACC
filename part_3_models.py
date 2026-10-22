"""Part 3/4: classification, numeric prediction and monthly forecasting.

Keep this file beside racc_streamlit_app.py. The main app imports it only when needed.
Analysis settings and output writers are bound to the current browser session.
"""
from __future__ import annotations

def build_modules(context):
    """Bind shared settings/services and return the task functions in this part."""
    COUNT_TEST_YEAR = context['COUNT_TEST_YEAR']
    ColumnTransformer = context['ColumnTransformer']
    DummyClassifier = context['DummyClassifier']
    DummyRegressor = context['DummyRegressor']
    ExtraTreesClassifier = context['ExtraTreesClassifier']
    FORECAST_HORIZON = context['FORECAST_HORIZON']
    HistGradientBoostingClassifier = context['HistGradientBoostingClassifier']
    HistGradientBoostingRegressor = context['HistGradientBoostingRegressor']
    INCLUDE_INVESTIGATED_CAUSE = context['INCLUDE_INVESTIGATED_CAUSE']
    LogisticRegression = context['LogisticRegression']
    OUTPUT_ROOT = context['OUTPUT_ROOT']
    OneHotEncoder = context['OneHotEncoder']
    Pipeline = context['Pipeline']
    PoissonRegressor = context['PoissonRegressor']
    RandomForestClassifier = context['RandomForestClassifier']
    RandomForestRegressor = context['RandomForestRegressor']
    Ridge = context['Ridge']
    SEED = context['SEED']
    SEVERITY_TEST_YEAR = context['SEVERITY_TEST_YEAR']
    SimpleImputer = context['SimpleImputer']
    StandardScaler = context['StandardScaler']
    accuracy_score = context['accuracy_score']
    binary_metrics = context['binary_metrics']
    clone = context['clone']
    count_metrics = context['count_metrics']
    display = context['display']
    f1_score = context['f1_score']
    math = context['math']
    np = context['np']
    optimize = context['optimize']
    output_helpers = context['output_helpers']
    pd = context['pd']
    plt = context['plt']
    print = context['print']
    warnings = context['warnings']

    def module_04_classification(state):
        """Train binary and four-class severity classifiers, select on development folds and generate holdout predictions.

        Inputs: Reliable injury-grade observations from Module 1.
        Returns: a dictionary of named module outputs."""
        OUTPUT_DIR = OUTPUT_ROOT / '04_classification'
        save_table, finish_figure = output_helpers(OUTPUT_DIR)
        injury_era = state['01_cleaning']['injury_era']

        calendar_features=['hour_sin','hour_cos','month_sin','month_cos','day_of_week_sin','day_of_week_cos','is_weekend']
        environment_features=['weather','road_condition','area']
        profile_features=['vehicle_group','gender_composition']
        FEATURES=calendar_features+environment_features+profile_features
        if INCLUDE_INVESTIGATED_CAUSE: FEATURES=FEATURES+['cause_group']
        numeric_features=calendar_features

        def classifier_pipeline(estimator,features=FEATURES):
            num=[c for c in features if c in numeric_features]
            cat=[c for c in features if c not in numeric_features]
            transformers=[]
            if num:
                transformers.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                                                        ('scale',StandardScaler())]),num))
            if cat:
                transformers.append(('categorical',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='missing')),
                    ('encode',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=10,sparse_output=False))]),cat))
            return Pipeline([('preprocess',ColumnTransformer(transformers,remainder='drop')),
                             ('model',clone(estimator))])

        # Convert pandas missing strings to NumPy NaN for consistent sklearn imputation.
        model_data=injury_era.sort_values(['date','source_row']).copy()
        for c in FEATURES:
            if c not in numeric_features:
                model_data[c]=model_data[c].astype(object).where(model_data[c].notna(),np.nan)
        dev=model_data.loc[model_data.year.lt(SEVERITY_TEST_YEAR)].copy()
        test=model_data.loc[model_data.year.eq(SEVERITY_TEST_YEAR)].copy()
        assert len(dev)>0 and len(test)>0
        assert dev.date.max()<test.date.min()
        assert set(dev.incident_proxy).isdisjoint(test.incident_proxy)
        X_dev,y_dev=dev[FEATURES],dev.severe.astype(int)
        X_test,y_test=test[FEATURES],test.severe.astype(int)
        severity_folds=[]
        for year in [2022,2023]:
            ti=np.flatnonzero(dev.year.to_numpy()<year);vi=np.flatnonzero(dev.year.to_numpy()==year)
            if len(ti) and len(vi):
                assert dev.iloc[ti].date.max()<dev.iloc[vi].date.min()
                assert set(dev.iloc[ti].incident_proxy).isdisjoint(dev.iloc[vi].incident_proxy)
                assert y_dev.iloc[ti].nunique()==2 and y_dev.iloc[vi].nunique()==2
                severity_folds.append((ti,vi))
        assert len(severity_folds)>=2, 'Need at least two valid development folds.'
        split_table=pd.DataFrame([{'split':'development','start':dev.date.min(),'end':dev.date.max(),'n':len(dev)},
                                 {'split':'holdout','start':test.date.min(),'end':test.date.max(),'n':len(test)}])
        display(save_table(split_table,'15_classification_split'))
        display(pd.crosstab(model_data.year,model_data.severity_grade))

        classifier_specs={
            'Prior baseline':DummyClassifier(strategy='prior'),
            'Logistic C=0.3':LogisticRegression(C=0.3,class_weight='balanced',max_iter=3000,random_state=SEED),
            'Logistic C=3':LogisticRegression(C=3,class_weight='balanced',max_iter=3000,random_state=SEED),
            'Random forest shallow':RandomForestClassifier(n_estimators=180,max_depth=5,min_samples_leaf=8,class_weight='balanced_subsample',random_state=SEED,n_jobs=1),
            'Random forest flexible':RandomForestClassifier(n_estimators=180,min_samples_leaf=5,class_weight='balanced_subsample',random_state=SEED,n_jobs=1),
            'Extra trees':ExtraTreesClassifier(n_estimators=180,max_depth=8,min_samples_leaf=5,class_weight='balanced',random_state=SEED,n_jobs=1),
            'Gradient boosting':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,l2_regularization=5,
                                 min_samples_leaf=15,early_stopping=False,class_weight='balanced',random_state=SEED)
        }

        cv_rows=[]
        for name,estimator in classifier_specs.items():
            for fold,(ti,vi) in enumerate(severity_folds,1):
                pipe=classifier_pipeline(estimator)
                pipe.fit(X_dev.iloc[ti],y_dev.iloc[ti])
                pred=pipe.predict(X_dev.iloc[vi]);prob=pipe.predict_proba(X_dev.iloc[vi])[:,1]
                cv_rows.append({'model':name,'fold':fold,**binary_metrics(y_dev.iloc[vi],pred,prob)})
        classifier_cv=pd.DataFrame(cv_rows)
        save_table(classifier_cv,'16_binary_classification_fold_scores')
        classifier_cv_summary=classifier_cv.groupby('model').agg(
            macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),
            balanced_accuracy_mean=('balanced_accuracy','mean'),ROC_AUC_mean=('ROC_AUC','mean'),
            severe_recall_mean=('severe_recall','mean'),Brier_mean=('Brier_score','mean')).sort_values('macro_f1_mean',ascending=False)
        display(save_table(classifier_cv_summary.reset_index(),'17_binary_classification_cv_summary'))
        best_classifier_name=classifier_cv_summary.index[0]  # Includes baseline: weak learned models need not win.
        best_classifier=classifier_pipeline(classifier_specs[best_classifier_name]).fit(X_dev,y_dev)
        print('Selected on development macro-F1:',best_classifier_name)

        test_pred=best_classifier.predict(X_test)
        test_prob=best_classifier.predict_proba(X_test)[:,1]
        prior_classifier=classifier_pipeline(DummyClassifier(strategy='prior')).fit(X_dev,y_dev)

        MC_LABELS=['no injury','minor injury','major injury','fatal']
        mc_candidates={k:classifier_specs[k] for k in ['Prior baseline','Logistic C=0.3','Random forest shallow']}
        mc_rows=[]
        for name,estimator in mc_candidates.items():
            for fold,(ti,vi) in enumerate(severity_folds,1):
                model=classifier_pipeline(estimator).fit(X_dev.iloc[ti],dev.severity_grade.iloc[ti])
                pred=model.predict(X_dev.iloc[vi])
                mc_rows.append({'model':name,'fold':fold,
                    'macro_f1':f1_score(dev.severity_grade.iloc[vi],pred,labels=MC_LABELS,average='macro',zero_division=0),
                    'accuracy':accuracy_score(dev.severity_grade.iloc[vi],pred)})
        mc_cv=pd.DataFrame(mc_rows)
        mc_summary=mc_cv.groupby('model')[['macro_f1','accuracy']].mean().sort_values('macro_f1',ascending=False)
        display(save_table(mc_summary.reset_index(),'22_multiclass_cv_summary'))
        mc_name=mc_summary.index[0]
        multiclass_model=classifier_pipeline(mc_candidates[mc_name]).fit(X_dev,dev.severity_grade)

        mc_pred=multiclass_model.predict(X_test)

        exported_names = ['FEATURES', 'numeric_features', 'calendar_features', 'environment_features', 'profile_features', 'classifier_pipeline', 'model_data', 'dev', 'test', 'X_dev', 'y_dev', 'X_test', 'y_test', 'severity_folds', 'classifier_specs', 'classifier_cv', 'classifier_cv_summary', 'best_classifier_name', 'best_classifier', 'prior_classifier', 'test_pred', 'test_prob', 'MC_LABELS', 'mc_candidates', 'mc_cv', 'mc_summary', 'mc_name', 'multiclass_model', 'mc_pred']
        return {name:value for name,value in locals().items() if name in exported_names}

    def module_05_numeric_prediction(state):
        """Build past-only daily features, compare regression models and baselines, and generate count predictions.

        Inputs: Daily record counts and the reporting calendar from Module 1.
        Returns: a dictionary of named module outputs."""
        OUTPUT_DIR = OUTPUT_ROOT / '05_numeric_prediction'
        save_table, finish_figure = output_helpers(OUTPUT_DIR)
        calendar = state['01_cleaning']['calendar']
        daily = state['01_cleaning']['daily']

        def make_daily_features(series):
            z=pd.DataFrame({'target':series.astype(float)})
            idx=z.index
            z['weekday_sin']=np.sin(2*np.pi*idx.dayofweek/7)
            z['weekday_cos']=np.cos(2*np.pi*idx.dayofweek/7)
            z['month_sin']=np.sin(2*np.pi*idx.month/12)
            z['month_cos']=np.cos(2*np.pi*idx.month/12)
            z['is_weekend']=(idx.dayofweek>=5).astype(int)
            z['trend_day']=(idx-calendar.min()).days
            for lag in [1,7,14,28]: z[f'lag_{lag}']=series.shift(lag)
            previous=series.shift(1)
            z['rolling_mean_7']=previous.rolling(7,min_periods=7).mean()
            z['rolling_mean_28']=previous.rolling(28,min_periods=28).mean()
            z['rolling_sd_7']=previous.rolling(7,min_periods=7).std()
            return z.iloc[28:].copy()  # Remove warm-up dates; do not fabricate prior history.

        daily_model=make_daily_features(daily.record_count)
        DAILY_FEATURES=[c for c in daily_model if c!='target']
        daily_dev=daily_model.loc[daily_model.index.year<COUNT_TEST_YEAR]
        daily_test=daily_model.loc[daily_model.index.year==COUNT_TEST_YEAR]
        assert daily_dev.index.max()<daily_test.index.min()
        assert np.allclose(daily_model['lag_1'],daily.record_count.shift(1).reindex(daily_model.index))
        daily_folds=[]
        for year in [2022,2023,2024]:
            ti=np.flatnonzero(daily_dev.index.year<year);vi=np.flatnonzero(daily_dev.index.year==year)
            if len(ti)>100 and len(vi)>30: daily_folds.append((ti,vi))
        assert len(daily_folds)>=2

        def regression_pipeline(estimator):
            return Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                             ('scale',StandardScaler()),('model',clone(estimator))])

        regressor_specs={
            'Mean-count baseline':DummyRegressor(strategy='mean'),
            'Median-count baseline':DummyRegressor(strategy='median'),
            'Ridge regression':Ridge(alpha=10),
            'Poisson alpha=0.1':PoissonRegressor(alpha=.1,max_iter=1500),
            'Poisson alpha=1':PoissonRegressor(alpha=1,max_iter=1500),
            'Poisson alpha=10':PoissonRegressor(alpha=10,max_iter=1500),
            'Random forest count':RandomForestRegressor(n_estimators=160,max_depth=6,min_samples_leaf=15,random_state=SEED,n_jobs=1),
            'Poisson gradient boosting':HistGradientBoostingRegressor(loss='poisson',max_iter=100,
                max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,early_stopping=False,random_state=SEED)
        }

        reg_rows=[]
        for name,estimator in regressor_specs.items():
            for fold,(ti,vi) in enumerate(daily_folds,1):
                pipe=regression_pipeline(estimator).fit(daily_dev[DAILY_FEATURES].iloc[ti],daily_dev.target.iloc[ti])
                pred=np.clip(pipe.predict(daily_dev[DAILY_FEATURES].iloc[vi]),0,None)
                reg_rows.append({'model':name,'fold':fold,**count_metrics(daily_dev.target.iloc[vi],pred)})
        regression_cv=pd.DataFrame(reg_rows)
        save_table(regression_cv,'29_daily_prediction_cv_fold_scores')
        regression_summary=regression_cv.groupby('model').agg(MAE_mean=('MAE','mean'),MAE_sd=('MAE','std'),
            RMSE_mean=('RMSE','mean'),Poisson_deviance_mean=('Poisson_deviance','mean')).sort_values('MAE_mean')
        display(save_table(regression_summary.reset_index(),'30_daily_prediction_cv_summary'))
        best_regressor_name=regression_summary.index[0]
        best_regressor=regression_pipeline(regressor_specs[best_regressor_name]).fit(daily_dev[DAILY_FEATURES],daily_dev.target)
        daily_pred=np.clip(best_regressor.predict(daily_test[DAILY_FEATURES]),0,None)
        daily_baseline=regression_pipeline(DummyRegressor(strategy='mean')).fit(daily_dev[DAILY_FEATURES],daily_dev.target)
        daily_baseline_pred=np.clip(daily_baseline.predict(daily_test[DAILY_FEATURES]),0,None)
        daily_seasonal_pred=daily_test.lag_7.to_numpy()  # Yesterday's week: legitimate one-day-ahead reference.
        learned_regressor_name=next(name for name in regression_summary.index if 'baseline' not in name.lower())
        learned_regressor=regression_pipeline(regressor_specs[learned_regressor_name]).fit(daily_dev[DAILY_FEATURES],daily_dev.target)
        learned_daily_pred=np.clip(learned_regressor.predict(daily_test[DAILY_FEATURES]),0,None)

        print('Selected on development MAE:',best_regressor_name)

        next_day=daily.index.max()+pd.Timedelta(days=1)
        history_with_next=pd.concat([daily.record_count,pd.Series([np.nan],index=[next_day])])
        next_features=make_daily_features(history_with_next).loc[[next_day],DAILY_FEATURES]
        daily_deployment_model=regression_pipeline(regressor_specs[best_regressor_name]).fit(
            daily_model[DAILY_FEATURES],daily_model.target)
        next_day_prediction=pd.DataFrame({'date':[next_day],
            'predicted_record_count':np.clip(daily_deployment_model.predict(next_features),0,None)})
        display(save_table(next_day_prediction,'34_next_day_prediction_after_dataset_end'))
        print('This date follows the dataset end; it may precede the current calendar date.')

        exported_names = ['make_daily_features', 'daily_model', 'DAILY_FEATURES', 'daily_dev', 'daily_test', 'daily_folds', 'regression_pipeline', 'regressor_specs', 'regression_cv', 'regression_summary', 'best_regressor_name', 'best_regressor', 'daily_pred', 'daily_baseline', 'daily_baseline_pred', 'daily_seasonal_pred', 'learned_regressor_name', 'learned_regressor', 'learned_daily_pred', 'daily_deployment_model', 'next_day_prediction']
        return {name:value for name,value in locals().items() if name in exported_names}

    def module_06_forecasting(state):
        """Backtest monthly forecasting methods, generate fixed-origin predictions and extrapolate beyond the dataset.

        Inputs: Monthly record-count series from Module 1.
        Returns: a dictionary of named module outputs."""
        OUTPUT_DIR = OUTPUT_ROOT / '06_forecasting'
        save_table, finish_figure = output_helpers(OUTPUT_DIR)
        monthly = state['01_cleaning']['monthly']

        def smooth_forecast(values,horizon,method):
            y=np.asarray(values,float)
            if method=='ses':
                def run(alpha):
                    level=y[0];errors=[]
                    for value in y[1:]:
                        errors.append(value-level);level=alpha*value+(1-alpha)*level
                    return level,np.asarray(errors)
                fit=optimize.minimize_scalar(lambda a:np.mean(run(a)[1]**2),bounds=(.001,.999),method='bounded')
                if not fit.success: raise RuntimeError('SES optimization failed.')
                return np.repeat(run(float(fit.x))[0],horizon)
            if method=='holt_damped':
                def run(parameters):
                    alpha,beta,phi=parameters;level=y[0]
                    trend=float(np.mean(np.diff(y[:min(6,len(y))])))
                    errors=[]
                    for value in y[1:]:
                        prediction=level+phi*trend;errors.append(value-prediction)
                        old_level=level;level=alpha*value+(1-alpha)*prediction
                        trend=beta*(level-old_level)+(1-beta)*phi*trend
                    return level,trend,np.asarray(errors)
                fits=[optimize.minimize(lambda p:np.mean(run(p)[2]**2),x0=start,
                    bounds=[(.001,.999),(.001,.999),(.8,1)],method='L-BFGS-B')
                    for start in [(0.2,0.1,0.9),(0.6,0.2,0.98)]]
                fit=min(fits,key=lambda z:z.fun)
                if not fit.success: warnings.warn('Holt optimizer did not fully converge; inspect fitted forecast.')
                level,trend,_=run(fit.x);phi=fit.x[2]
                return np.array([level+trend*np.sum(phi**np.arange(1,h+1)) for h in range(1,horizon+1)])
            raise ValueError(method)

        def monthly_lag_features(series):
            z=pd.DataFrame({'target':series.astype(float)})
            z['month_sin']=np.sin(2*np.pi*z.index.month/12)
            z['month_cos']=np.cos(2*np.pi*z.index.month/12)
            z['trend_month']=(z.index.year-2021)*12+z.index.month
            for lag in [1,3,12]: z[f'lag_{lag}']=series.shift(lag)
            z['rolling_3']=series.shift(1).rolling(3).mean()
            z['rolling_12']=series.shift(1).rolling(12).mean()
            return z.iloc[12:]

        def forecast_monthly(series,horizon,method):
            y=np.asarray(series,float)
            if len(y)<24: raise ValueError('At least 24 monthly training observations required.')
            if method=='last_value': pred=np.repeat(y[-1],horizon)
            elif method=='seasonal_naive': pred=np.array([y[-12+(i%12)] for i in range(horizon)])
            elif method=='drift': pred=y[-1]+np.arange(1,horizon+1)*(y[-1]-y[0])/(len(y)-1)
            elif method in ['ses','holt_damped']: pred=smooth_forecast(y,horizon,method)
            elif method=='recursive_random_forest':
                train=monthly_lag_features(series)
                cols=[c for c in train if c!='target']
                model=RandomForestRegressor(n_estimators=140,max_depth=3,min_samples_leaf=3,random_state=SEED,n_jobs=1)
                model.fit(train[cols],train.target)
                history=series.copy().astype(float);pred=[]
                for _ in range(horizon):
                    date=history.index[-1]+pd.offsets.MonthBegin(1)
                    augmented=pd.concat([history,pd.Series([np.nan],index=[date])])
                    value=float(np.clip(model.predict(monthly_lag_features(augmented).loc[[date],cols])[0],0,None))
                    pred.append(value);history.loc[date]=value  # Append prediction, never future actuals.
            else: raise ValueError(method)
            return np.clip(np.asarray(pred,float),0,None)

        FORECAST_METHODS=['last_value','seasonal_naive','drift','ses','holt_damped','recursive_random_forest']
        monthly_series=monthly.record_count.astype(float)
        forecast_dev=monthly_series.loc[monthly_series.index.year<COUNT_TEST_YEAR]
        forecast_test=monthly_series.loc[monthly_series.index.year==COUNT_TEST_YEAR]
        assert len(forecast_dev)==48 and len(forecast_test)==12, 'Review hard-coded calendar/backtest windows for a new dataset.'
        monthly_backtest_rows=[];monthly_backtest_predictions=[]
        for origin in [24,36]:
            train=forecast_dev.iloc[:origin];validation=forecast_dev.iloc[origin:origin+12]
            assert train.index.max()<validation.index.min()
            for method in FORECAST_METHODS:
                pred=forecast_monthly(train,len(validation),method)
                monthly_backtest_rows.append({'model':method,'train_months':len(train),
                    'validation_year':validation.index[0].year,**count_metrics(validation,pred)})
                monthly_backtest_predictions.extend({'model':method,'origin':str(train.index[-1].date()),
                    'date':date,'horizon':h+1,'actual':float(a),'predicted':float(p),'error':float(a-p)}
                    for h,(date,a,p) in enumerate(zip(validation.index,validation,pred)))
        forecast_backtests=pd.DataFrame(monthly_backtest_rows)
        forecast_backtest_predictions=pd.DataFrame(monthly_backtest_predictions)
        save_table(forecast_backtests,'35_monthly_forecast_backtest_scores')
        save_table(forecast_backtest_predictions,'36_monthly_forecast_backtest_predictions')
        forecast_cv_summary=forecast_backtests.groupby('model').agg(MAE_mean=('MAE','mean'),MAE_sd=('MAE','std'),
            RMSE_mean=('RMSE','mean'),WAPE_pct_mean=('WAPE_pct','mean')).sort_values('MAE_mean')
        display(save_table(forecast_cv_summary.reset_index(),'37_monthly_forecast_cv_summary'))
        best_forecast_method=forecast_cv_summary.index[0]
        print('Selected on development MAE:',best_forecast_method)

        monthly_test_pred=forecast_monthly(forecast_dev,len(forecast_test),best_forecast_method)
        seasonal_test_pred=forecast_monthly(forecast_dev,len(forecast_test),'seasonal_naive')
        naive_test_pred=forecast_monthly(forecast_dev,len(forecast_test),'last_value')

        development_errors=np.abs(forecast_backtest_predictions.loc[
            forecast_backtest_predictions.model.eq(best_forecast_method),'error'].to_numpy())
        INTERVAL_ALPHA=.10
        adjusted_quantile=min(1,math.ceil((len(development_errors)+1)*(1-INTERVAL_ALPHA))/len(development_errors))
        interval_half_width=float(np.quantile(development_errors,adjusted_quantile,method='higher'))
        monthly_test_lower=np.maximum(0,monthly_test_pred-interval_half_width)
        monthly_test_upper=monthly_test_pred+interval_half_width

        future_index=pd.date_range(monthly_series.index[-1]+pd.offsets.MonthBegin(1),periods=FORECAST_HORIZON,freq='MS')
        future_pred=forecast_monthly(monthly_series,FORECAST_HORIZON,best_forecast_method)
        future_forecast=pd.DataFrame({'date':future_index,'forecast_expected_record_count':future_pred,
            'approx_90pct_lower':np.maximum(0,future_pred-interval_half_width),
            'approx_90pct_upper':future_pred+interval_half_width})
        display(save_table(future_forecast,'40_forecast_after_dataset_end'))
        plt.figure(figsize=(13,5));plt.plot(monthly_series.index,monthly_series,label='Observed reported rows')
        plt.plot(future_index,future_pred,'o-',label=f'Forecast: {best_forecast_method}')
        plt.fill_between(future_index,future_forecast.approx_90pct_lower,future_forecast.approx_90pct_upper,alpha=.2,
                         label='Approximate 90% interval')
        plt.axvline(future_index[0],color='gray',ls='--');plt.ylabel('Recorded rows');plt.legend()
        plt.title('Forecast beyond the dataset end; refresh the data before current operational use')
        finish_figure('16_forecast_after_dataset_end')

        exported_names = ['smooth_forecast', 'monthly_lag_features', 'forecast_monthly', 'FORECAST_METHODS', 'monthly_series', 'forecast_dev', 'forecast_test', 'forecast_backtests', 'forecast_backtest_predictions', 'forecast_cv_summary', 'best_forecast_method', 'monthly_test_pred', 'seasonal_test_pred', 'naive_test_pred', 'development_errors', 'interval_half_width', 'monthly_test_lower', 'monthly_test_upper', 'future_index', 'future_pred', 'future_forecast']
        return {name:value for name,value in locals().items() if name in exported_names}

    return {'04_classification': module_04_classification, '05_prediction': module_05_numeric_prediction, '06_forecasting': module_06_forecasting}
