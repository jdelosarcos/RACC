"""Part 4/4: model evaluation, feature combinations and downloadable exports.

Keep this file beside racc_streamlit_app.py. The main app imports it only when needed.
Analysis settings and output writers are bound to the current browser session.
"""
from __future__ import annotations

def build_modules(context):
    """Bind shared settings/services and return the task functions in this part."""
    APPLY_CONTEXT_DATE_FIXES = context['APPLY_CONTEXT_DATE_FIXES']
    ASSUME_COMPLETE_REPORTING_CALENDAR = context['ASSUME_COMPLETE_REPORTING_CALENDAR']
    COUNT_TEST_YEAR = context['COUNT_TEST_YEAR']
    ConfusionMatrixDisplay = context['ConfusionMatrixDisplay']
    DROP_EXACT_DUPLICATES = context['DROP_EXACT_DUPLICATES']
    DecisionTreeClassifier = context['DecisionTreeClassifier']
    FORECAST_HORIZON = context['FORECAST_HORIZON']
    INCLUDE_INVESTIGATED_CAUSE = context['INCLUDE_INVESTIGATED_CAUSE']
    MIN_COMBINATION_RECORDS = context['MIN_COMBINATION_RECORDS']
    MODULE_SPECS = context['MODULE_SPECS']
    Markdown = context['Markdown']
    N_BOOTSTRAP = context['N_BOOTSTRAP']
    N_IMPORTANCE_REPEATS = context['N_IMPORTANCE_REPEATS']
    OUTPUT_ROOT = context['OUTPUT_ROOT']
    Path = context['Path']
    PoissonRegressor = context['PoissonRegressor']
    PrecisionRecallDisplay = context['PrecisionRecallDisplay']
    RocCurveDisplay = context['RocCurveDisplay']
    SEED = context['SEED']
    SEVERITY_TEST_YEAR = context['SEVERITY_TEST_YEAR']
    accuracy_score = context['accuracy_score']
    binary_metrics = context['binary_metrics']
    calibration_curve = context['calibration_curve']
    classification_report = context['classification_report']
    count_metrics = context['count_metrics']
    display = context['display']
    export_text = context['export_text']
    f1_score = context['f1_score']
    joblib = context['joblib']
    json = context['json']
    mean_absolute_error = context['mean_absolute_error']
    np = context['np']
    output_helpers = context['output_helpers']
    pd = context['pd']
    permutation_importance = context['permutation_importance']
    plt = context['plt']
    print = context['print']
    roc_auc_score = context['roc_auc_score']
    shutil = context['shutil']
    versions = context['versions']

    def module_07_model_performance(state):
        """Evaluate the selected models on reserved years and report task-specific metrics, diagnostics and uncertainty.

        Inputs: Fitted development models, untouched holdout predictions and count series from Modules 1, 4, 5 and 6.
        Returns: a dictionary of named module outputs."""
        OUTPUT_DIR = OUTPUT_ROOT / '07_model_performance'
        save_table, finish_figure = output_helpers(OUTPUT_DIR)
        DAILY_FEATURES = state['05_prediction']['DAILY_FEATURES']
        FEATURES = state['04_classification']['FEATURES']
        MC_LABELS = state['04_classification']['MC_LABELS']
        X_test = state['04_classification']['X_test']
        best_classifier_name = state['04_classification']['best_classifier_name']
        best_forecast_method = state['06_forecasting']['best_forecast_method']
        best_regressor_name = state['05_prediction']['best_regressor_name']
        daily = state['01_cleaning']['daily']
        daily_baseline_pred = state['05_prediction']['daily_baseline_pred']
        daily_pred = state['05_prediction']['daily_pred']
        daily_seasonal_pred = state['05_prediction']['daily_seasonal_pred']
        daily_test = state['05_prediction']['daily_test']
        dev = state['04_classification']['dev']
        development_errors = state['06_forecasting']['development_errors']
        forecast_dev = state['06_forecasting']['forecast_dev']
        forecast_monthly = state['06_forecasting']['forecast_monthly']
        forecast_test = state['06_forecasting']['forecast_test']
        future_pred = state['06_forecasting']['future_pred']
        learned_daily_pred = state['05_prediction']['learned_daily_pred']
        learned_regressor_name = state['05_prediction']['learned_regressor_name']
        make_daily_features = state['05_prediction']['make_daily_features']
        mc_name = state['04_classification']['mc_name']
        mc_pred = state['04_classification']['mc_pred']
        monthly = state['01_cleaning']['monthly']
        monthly_test_lower = state['06_forecasting']['monthly_test_lower']
        monthly_test_pred = state['06_forecasting']['monthly_test_pred']
        monthly_test_upper = state['06_forecasting']['monthly_test_upper']
        naive_test_pred = state['06_forecasting']['naive_test_pred']
        prior_classifier = state['04_classification']['prior_classifier']
        seasonal_test_pred = state['06_forecasting']['seasonal_test_pred']
        severity_folds = state['04_classification']['severity_folds']
        test = state['04_classification']['test']
        test_pred = state['04_classification']['test_pred']
        test_prob = state['04_classification']['test_prob']
        y_test = state['04_classification']['y_test']

        classification_holdout=pd.DataFrame([
            {'model':best_classifier_name,**binary_metrics(y_test,test_pred,test_prob)},
            {'model':'Prior baseline (reference)',**binary_metrics(y_test,prior_classifier.predict(X_test),prior_classifier.predict_proba(X_test)[:,1])}])
        display(save_table(classification_holdout,'18_binary_classification_holdout_metrics'))
        binary_report=pd.DataFrame(classification_report(y_test,test_pred,labels=[0,1],
            target_names=['non-serious recorded grade','major / fatal'],output_dict=True,zero_division=0)).T
        display(save_table(binary_report.reset_index(names='class_or_average'),'19_binary_classification_report'))
        holdout_predictions=test[['source_row','date','incident_proxy','severity_grade']].copy()
        holdout_predictions['actual_severe']=y_test.to_numpy();holdout_predictions['predicted_severe']=test_pred
        holdout_predictions['probability_major_or_fatal']=test_prob
        save_table(holdout_predictions,'20_binary_holdout_predictions')

        fig,axes=plt.subplots(1,3,figsize=(16,4.5))
        ConfusionMatrixDisplay.from_predictions(y_test,test_pred,display_labels=['non-serious','major/fatal'],ax=axes[0],colorbar=False)
        RocCurveDisplay.from_predictions(y_test,test_prob,ax=axes[1],name='Selected model')
        axes[1].plot([0,1],[0,1],'k--',alpha=.5)
        PrecisionRecallDisplay.from_predictions(y_test,test_prob,ax=axes[2],name='Selected model')
        axes[2].axhline(y_test.mean(),ls='--',color='gray',label='Severe prevalence');axes[2].legend()
        finish_figure('08_binary_classifier_diagnostics')

        observed,predicted=calibration_curve(y_test,test_prob,n_bins=6,strategy='quantile')
        plt.plot(predicted,observed,'o-',label=best_classifier_name);plt.plot([0,1],[0,1],'k--',label='Perfect calibration')
        plt.xlabel('Mean predicted probability');plt.ylabel('Observed severe fraction');plt.legend()
        plt.title('Holdout calibration check; class weighting does not guarantee calibrated probabilities')
        finish_figure('09_probability_calibration')

        # Resample whole provisional incidents, keeping all associated test rows together.
        group_indices=[np.flatnonzero(test.incident_proxy.to_numpy()==g) for g in test.incident_proxy.unique()]
        bootstrap_rows=[];boot_rng=np.random.default_rng(SEED)
        for _ in range(N_BOOTSTRAP):
            idx=np.concatenate([group_indices[i] for i in boot_rng.integers(0,len(group_indices),len(group_indices))])
            yy=y_test.to_numpy()[idx];pp=test_pred[idx];pr=test_prob[idx]
            bootstrap_rows.append({'accuracy':accuracy_score(yy,pp),'macro_f1':f1_score(yy,pp,average='macro',zero_division=0),
                                  'ROC_AUC':roc_auc_score(yy,pr) if len(np.unique(yy))==2 else np.nan})
        bootstrap_ci=pd.DataFrame(bootstrap_rows).quantile([.025,.975]).T.rename(columns={.025:'95pct_low',.975:'95pct_high'})
        display(save_table(bootstrap_ci.reset_index(names='metric'),'21_incident_proxy_bootstrap_intervals'))
        print('Intervals condition on the chosen model, split and proxy grouping; they do not include model-selection uncertainty.')

        mc_report=pd.DataFrame(classification_report(test.severity_grade,mc_pred,labels=MC_LABELS,
                                                   output_dict=True,zero_division=0)).T
        display(save_table(mc_report.reset_index(names='class_or_average'),'23_multiclass_holdout_report'))
        ConfusionMatrixDisplay.from_predictions(test.severity_grade,mc_pred,labels=MC_LABELS,
                                                xticks_rotation=25,colorbar=False)
        plt.title(f'Four-class holdout: {mc_name}');finish_figure('10_multiclass_confusion_matrix')

        daily_holdout=pd.DataFrame([
            {'model':best_regressor_name,**count_metrics(daily_test.target,daily_pred)},
            {'model':learned_regressor_name+' (best learned candidate by development MAE)',**count_metrics(daily_test.target,learned_daily_pred)},
            {'model':'Mean-count baseline (reference)',**count_metrics(daily_test.target,daily_baseline_pred)},
            {'model':'Same weekday previous week',**count_metrics(daily_test.target,daily_seasonal_pred)}])
        display(save_table(daily_holdout,'31_daily_prediction_holdout_metrics'))
        daily_predictions=pd.DataFrame({'date':daily_test.index,'actual_record_count':daily_test.target,
            'predicted_record_count':daily_pred,'best_learned_candidate_prediction':learned_daily_pred,'mean_baseline':daily_baseline_pred,
            'previous_week_baseline':daily_seasonal_pred}).reset_index(drop=True)
        save_table(daily_predictions,'32_daily_holdout_predictions')

        fig,axes=plt.subplots(1,2,figsize=(14,4.5))
        axes[0].plot(daily_test.index,daily_test.target,label='Observed',alpha=.6,lw=.8)
        axes[0].plot(daily_test.index,daily_pred,label='Selected one-day-ahead point prediction',lw=1)
        axes[0].set(title='Daily holdout predictions',ylabel='Recorded rows');axes[0].legend()
        residual=daily_test.target.to_numpy()-daily_pred
        axes[1].scatter(daily_pred,residual,alpha=.3,s=15)
        axes[1].axhline(0,color='black',lw=.8)
        axes[1].set(xlabel='Predicted count',ylabel='Actual minus predicted',title='Holdout residuals')
        finish_figure('12_daily_prediction_diagnostics')

        # A constant baseline cannot attribute predictions to variables. If it wins, inspect the
        # strongest learned candidate separately, without replacing the selected baseline.

        mase_scale=np.mean(np.abs(forecast_dev.to_numpy()[12:]-forecast_dev.to_numpy()[:-12]))
        forecast_holdout_rows=[]
        for name,pred in [(best_forecast_method,monthly_test_pred),('seasonal_naive (reference)',seasonal_test_pred),
                          ('last_value (reference)',naive_test_pred)]:
            row={'model':name,**count_metrics(forecast_test,pred)}
            row['seasonal_MASE']=row['MAE']/mase_scale if mase_scale>0 else np.nan
            forecast_holdout_rows.append(row)
        forecast_holdout=pd.DataFrame(forecast_holdout_rows)
        display(save_table(forecast_holdout,'38_monthly_forecast_holdout_metrics'))

        interval_coverage=float(np.mean((forecast_test.to_numpy()>=monthly_test_lower)&(forecast_test.to_numpy()<=monthly_test_upper)))
        print(f'Approximate 90% interval: {len(development_errors)} pooled development errors; holdout coverage={interval_coverage:.1%}')
        monthly_holdout_predictions=pd.DataFrame({'date':forecast_test.index,'actual_record_count':forecast_test.to_numpy(),
            'predicted_expected_count':monthly_test_pred,'approx_90pct_lower':monthly_test_lower,
            'approx_90pct_upper':monthly_test_upper,'seasonal_naive':seasonal_test_pred})
        save_table(monthly_holdout_predictions,'39_monthly_holdout_predictions')

        fig,ax=plt.subplots(figsize=(13,5))
        ax.plot(forecast_dev.index,forecast_dev,label='Development observations',color='#4b6570')
        ax.plot(forecast_test.index,forecast_test,label='Holdout actual',color='black',marker='o')
        ax.plot(forecast_test.index,monthly_test_pred,label=f'Fixed-origin forecast: {best_forecast_method}',marker='o')
        ax.fill_between(forecast_test.index,monthly_test_lower,monthly_test_upper,alpha=.2,label='Approximate 90% interval')
        ax.axvline(forecast_test.index[0],ls='--',color='gray');ax.set(ylabel='Reported rows',title='12-month fixed-origin forecast evaluation')
        ax.legend(loc='upper left',fontsize=9);finish_figure('14_monthly_forecast_holdout')

        monthly_residual=forecast_test.to_numpy()-monthly_test_pred
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        axes[0].bar(forecast_test.index,monthly_residual,width=20);axes[0].axhline(0,color='black',lw=.8)
        axes[0].set(title='Monthly holdout residuals',ylabel='Actual minus forecast')
        lags=np.arange(1,6)
        autocorrelations=[pd.Series(monthly_residual).autocorr(lag=int(k)) for k in lags]
        axes[1].stem(lags,autocorrelations);axes[1].axhline(0,color='black',lw=.8)
        axes[1].set(xlabel='Lag in months',ylabel='Residual autocorrelation',title='Descriptive only: just 12 holdout months')
        finish_figure('15_monthly_forecast_residuals')

        sensitivity_rows=[]
        for col in ['record_count','proxy_count','strict_date_record_count']:
            s=monthly[col].astype(float);train=s.loc[s.index.year<COUNT_TEST_YEAR];actual=s.loc[s.index.year==COUNT_TEST_YEAR]
            pred=forecast_monthly(train,len(actual),best_forecast_method)
            sensitivity_rows.append({'count_definition':col,'all_year_total':int(s.sum()),'holdout_total':int(actual.sum()),
                'fixed_method':best_forecast_method,**count_metrics(actual,pred)})
        sensitivity_summary=pd.DataFrame(sensitivity_rows)
        display(save_table(sensitivity_summary,'41_date_and_count_definition_sensitivity'))

        # Demonstrate that count feature construction cannot see the current/future target.
        probe_date=pd.Timestamp('2025-06-15')
        perturbed=daily.record_count.copy();perturbed.loc[probe_date:]+=100
        original_features=make_daily_features(daily.record_count).loc[probe_date,DAILY_FEATURES]
        perturbed_features=make_daily_features(perturbed).loc[probe_date,DAILY_FEATURES]
        assert np.allclose(original_features,perturbed_features,equal_nan=True), 'Current/future count leaked into predictor.'
        assert 'severe' not in FEATURES and 'severity_grade' not in FEATURES and 'incident_proxy' not in FEATURES
        assert not set(dev.incident_proxy).intersection(test.incident_proxy)
        assert all(dev.iloc[ti].date.max()<dev.iloc[vi].date.min() for ti,vi in severity_folds)
        assert np.isfinite(daily_pred).all() and np.isfinite(future_pred).all()
        assert (daily_pred>=0).all() and (future_pred>=0).all()
        print('Checks passed: chronological separation, grouped records, target exclusion, past-only daily predictors and finite nonnegative forecasts.')

        performance_dashboard=pd.DataFrame([
            {'task':'binary severity classification','selected_model':best_classifier_name,
             'holdout_year':SEVERITY_TEST_YEAR,'metric':'macro-F1','selected_value':classification_holdout.iloc[0]['macro_f1'],
             'reference_value':classification_holdout.iloc[1]['macro_f1'],'larger_is_better':True},
            {'task':'one-day-ahead count prediction','selected_model':best_regressor_name,
             'holdout_year':COUNT_TEST_YEAR,'metric':'MAE','selected_value':daily_holdout.iloc[0]['MAE'],
             'reference_value':daily_holdout.loc[daily_holdout.model.eq('Mean-count baseline (reference)'),'MAE'].iloc[0],
             'larger_is_better':False},
            {'task':'12-month fixed-origin forecasting','selected_model':best_forecast_method,
             'holdout_year':COUNT_TEST_YEAR,'metric':'MAE','selected_value':forecast_holdout.iloc[0]['MAE'],
             'reference_value':forecast_holdout.iloc[1]['MAE'],'larger_is_better':False}
        ])
        display(save_table(performance_dashboard,'43_task_performance_dashboard'))

        exported_names = ['classification_holdout', 'binary_report', 'holdout_predictions', 'bootstrap_ci', 'mc_report', 'daily_holdout', 'daily_predictions', 'forecast_holdout', 'interval_coverage', 'monthly_holdout_predictions', 'sensitivity_summary', 'performance_dashboard']
        return {name:value for name,value in locals().items() if name in exported_names}

    def module_08_feature_analysis(state):
        """Inspect original-variable importance, predefined feature combinations, supported observed patterns, rules and scenarios.

        Inputs: Cleaned data, fitted models, temporal folds and evaluation context from Modules 1, 4, 5 and 7.
        Returns: a dictionary of named module outputs."""
        OUTPUT_DIR = OUTPUT_ROOT / '08_features_and_combinations'
        save_table, finish_figure = output_helpers(OUTPUT_DIR)
        DAILY_FEATURES = state['05_prediction']['DAILY_FEATURES']
        FEATURES = state['04_classification']['FEATURES']
        X_dev = state['04_classification']['X_dev']
        X_test = state['04_classification']['X_test']
        best_classifier = state['04_classification']['best_classifier']
        best_classifier_name = state['04_classification']['best_classifier_name']
        best_regressor = state['05_prediction']['best_regressor']
        best_regressor_name = state['05_prediction']['best_regressor_name']
        calendar_features = state['04_classification']['calendar_features']
        classifier_pipeline = state['04_classification']['classifier_pipeline']
        classifier_specs = state['04_classification']['classifier_specs']
        clean_records = state['01_cleaning']['clean_records']
        daily_dev = state['05_prediction']['daily_dev']
        daily_folds = state['05_prediction']['daily_folds']
        daily_test = state['05_prediction']['daily_test']
        dev = state['04_classification']['dev']
        environment_features = state['04_classification']['environment_features']
        learned_regressor = state['05_prediction']['learned_regressor']
        learned_regressor_name = state['05_prediction']['learned_regressor_name']
        numeric_features = state['04_classification']['numeric_features']
        profile_features = state['04_classification']['profile_features']
        regression_pipeline = state['05_prediction']['regression_pipeline']
        regressor_specs = state['05_prediction']['regressor_specs']
        severity_folds = state['04_classification']['severity_folds']
        y_dev = state['04_classification']['y_dev']
        y_test = state['04_classification']['y_test']

        importance=permutation_importance(best_classifier,X_test,y_test,scoring='f1_macro',
                                        n_repeats=N_IMPORTANCE_REPEATS,random_state=SEED,n_jobs=1)
        severity_importance=pd.DataFrame({'feature':FEATURES,'macro_f1_drop_mean':importance.importances_mean,
                                        'repeat_sd':importance.importances_std}).sort_values('macro_f1_drop_mean',ascending=False)
        display(save_table(severity_importance,'24_severity_permutation_importance'))
        plot_importance=severity_importance.sort_values('macro_f1_drop_mean')
        plt.barh(plot_importance.feature,plot_importance.macro_f1_drop_mean,xerr=plot_importance.repeat_sd,color='#147d92')
        plt.axvline(0,color='black',lw=.8);plt.xlabel('Holdout macro-F1 decrease after permutation')
        plt.title('Severity model: original-variable importance (repeat SD bars)')
        finish_figure('11_severity_variable_importance')

        feature_sets={'calendar':calendar_features,'environment':environment_features,'recorded vehicle / gender':profile_features,
            'calendar + environment':calendar_features+environment_features,
            'environment + vehicle / gender':environment_features+profile_features,'all configured features':FEATURES}
        ablation_rows=[]
        for name,cols in feature_sets.items():
            scores=[]
            for ti,vi in severity_folds:
                p=classifier_pipeline(classifier_specs[best_classifier_name],cols).fit(dev[cols].iloc[ti],y_dev.iloc[ti])
                scores.append(f1_score(y_dev.iloc[vi],p.predict(dev[cols].iloc[vi]),average='macro',zero_division=0))
            ablation_rows.append({'feature_combination':name,'n_original_variables':len(cols),
                'variables':', '.join(cols),'CV_macro_f1_mean':np.mean(scores),'CV_macro_f1_sd':np.std(scores,ddof=1)})
        ablation_results=pd.DataFrame(ablation_rows).sort_values('CV_macro_f1_mean',ascending=False)
        display(save_table(ablation_results,'25_feature_combination_ablation'))

        oof_frames=[]
        for ti,vi in severity_folds:
            p=classifier_pipeline(classifier_specs[best_classifier_name]).fit(X_dev.iloc[ti],y_dev.iloc[ti])
            part=dev.iloc[vi][['weather','road_condition','vehicle_group','incident_proxy','severe']].copy()
            part['oof_probability']=p.predict_proba(X_dev.iloc[vi])[:,1]
            oof_frames.append(part)
        oof=pd.concat(oof_frames,ignore_index=True)
        combinations=oof.groupby(['weather','road_condition','vehicle_group'],dropna=False).agg(
            records=('severe','size'),unique_proxy_support=('incident_proxy','nunique'),
            observed_severe_record_fraction=('severe','mean'),mean_out_of_fold_probability=('oof_probability','mean')).reset_index()
        combinations=combinations.loc[combinations.records.ge(MIN_COMBINATION_RECORDS)].sort_values('records',ascending=False)
        display(save_table(combinations,'26_supported_weather_road_vehicle_combinations'))

        rule_model=classifier_pipeline(DecisionTreeClassifier(max_depth=3,min_samples_leaf=30,random_state=SEED)).fit(X_dev,y_dev)
        rule_names=rule_model.named_steps['preprocess'].get_feature_names_out()
        rule_text=export_text(rule_model.named_steps['model'],feature_names=list(rule_names),decimals=2)
        print(rule_text)
        (OUTPUT_DIR/'severity_reference_tree_rules.txt').write_text(rule_text,encoding='utf-8')
        tree=rule_model.named_steps['model'].tree_;tree_classes=rule_model.named_steps['model'].classes_
        rule_rows=[]
        def walk_rules(node=0,conditions=None):
            conditions=[] if conditions is None else conditions
            if tree.children_left[node]==tree.children_right[node]:
                values=tree.value[node][0];prob=values/values.sum()
                rule_rows.append({'conditions':' AND '.join(conditions) or '(root)',
                    'development_record_support':int(tree.n_node_samples[node]),
                    'tree_probability_severe':float(prob[list(tree_classes).index(1)]),
                    'predicted_class':int(tree_classes[np.argmax(prob)])})
                return
            name=rule_names[tree.feature[node]];threshold=tree.threshold[node]
            walk_rules(tree.children_left[node],conditions+[f'{name} <= {threshold:.3f}'])
            walk_rules(tree.children_right[node],conditions+[f'{name} > {threshold:.3f}'])
        walk_rules()
        display(save_table(pd.DataFrame(rule_rows),'27_reference_decision_tree_combinations'))
        print('Reference tree holdout macro-F1:',round(f1_score(y_test,rule_model.predict(X_test),average='macro'),3))

        def predict_record_severity(record,model=best_classifier):
            # Supply original CSV column names; deterministic cleaning matches the training notebook.
            incoming=clean_records(pd.DataFrame([record]))
            for c in FEATURES:
                if c not in numeric_features:
                    incoming[c]=incoming[c].astype(object).where(incoming[c].notna(),np.nan)
            if incoming.date.isna().any():
                raise ValueError('Supply a usable DATE for calendar features.')
            p=float(model.predict_proba(incoming[FEATURES])[0,1])
            return {'predicted_class':int(model.predict(incoming[FEATURES])[0]),
                    'probability_major_or_fatal':p,'model':best_classifier_name,
                    'cleaned_area':incoming.area.iloc[0],'vehicle_group':incoming.vehicle_group.iloc[0]}

        example={'DATE':'15/08/2024','TIME':'8:00 pm','LOCATION':'P1 San Teodoro',
                 'WEATHER':'Rainy','ROAD CONDITION':'Wet','VEHICLE TYPE':'Motorcycle','GENDER':'Male'}
        display(pd.DataFrame([predict_record_severity(example)]))
        scenario_rows=[]
        for weather,road in [('Clear','Dry'),('Rainy','Wet'),('Cloudy','Dry')]:
            rec={**example,'WEATHER':weather,'ROAD CONDITION':road}
            scenario_rows.append({'weather':weather,'road_condition':road,**predict_record_severity(rec)})
        display(save_table(pd.DataFrame(scenario_rows),'28_illustrative_prediction_scenarios'))
        print('Illustrative model scenarios; not causal interventions or calibrated operational risk estimates.')

        inspection_regressor=learned_regressor if 'baseline' in best_regressor_name.lower() else best_regressor
        inspection_regressor_name=learned_regressor_name if 'baseline' in best_regressor_name.lower() else best_regressor_name
        print('Daily feature importance describes:',inspection_regressor_name)
        daily_pi=permutation_importance(inspection_regressor,daily_test[DAILY_FEATURES],daily_test.target,
            scoring='neg_mean_absolute_error',n_repeats=N_IMPORTANCE_REPEATS,random_state=SEED,n_jobs=1)
        daily_importance=pd.DataFrame({'feature':DAILY_FEATURES,'MAE_increase_mean':daily_pi.importances_mean,
                                     'repeat_sd':daily_pi.importances_std}).sort_values('MAE_increase_mean',ascending=False)
        daily_importance['inspected_model']=inspection_regressor_name
        display(save_table(daily_importance,'33_daily_prediction_variable_importance'))
        p=daily_importance.sort_values('MAE_increase_mean')
        plt.barh(p.feature,p.MAE_increase_mean,xerr=p.repeat_sd,color='#6e638f')
        plt.axvline(0,color='black',lw=.8);plt.xlabel('Holdout MAE increase after permutation')
        plt.title(f'Daily count variable importance: {inspection_regressor_name}')
        finish_figure('13_daily_prediction_variable_importance')

        # Next-day prediction after the last observed dataset date, using only known past counts.

        daily_calendar=[c for c in DAILY_FEATURES if c.startswith(('weekday','month')) or c in ['is_weekend','trend_day']]
        daily_lags=[c for c in DAILY_FEATURES if c.startswith('lag_')]
        daily_rolling=[c for c in DAILY_FEATURES if c.startswith('rolling_')]
        daily_feature_sets={'calendar only':daily_calendar,'past lags only':daily_lags,
            'rolling summaries only':daily_rolling,'calendar + lags':daily_calendar+daily_lags,
            'calendar + rolling summaries':daily_calendar+daily_rolling,'all daily features':DAILY_FEATURES}
        daily_ablation_rows=[]
        for name,cols in daily_feature_sets.items():
            scores=[]
            for ti,vi in daily_folds:
                model=regression_pipeline(regressor_specs[learned_regressor_name]).fit(
                    daily_dev[cols].iloc[ti],daily_dev.target.iloc[ti])
                pred=np.clip(model.predict(daily_dev[cols].iloc[vi]),0,None)
                scores.append(mean_absolute_error(daily_dev.target.iloc[vi],pred))
            daily_ablation_rows.append({'feature_combination':name,'inspected_model':learned_regressor_name,
                'variables':', '.join(cols),'CV_MAE_mean':np.mean(scores),'CV_MAE_sd':np.std(scores,ddof=1)})
        daily_ablation=pd.DataFrame(daily_ablation_rows).sort_values('CV_MAE_mean')
        display(save_table(daily_ablation,'33b_daily_feature_combination_ablation'))
        if isinstance(learned_regressor.named_steps['model'],PoissonRegressor):
            coefficients=learned_regressor.named_steps['model'].coef_
            scale=learned_regressor.named_steps['scale'].scale_
            poisson_coefficients=pd.DataFrame({'feature':DAILY_FEATURES,'standardized_log_count_coefficient':coefficients,
                'log_count_coefficient_per_raw_unit':coefficients/scale,
                'count_ratio_per_raw_unit':np.exp(coefficients/scale)})
            display(save_table(poisson_coefficients,'33c_poisson_count_coefficients'))
            print('Coefficients describe the fitted log-link model holding other inputs fixed; no causal or significance claim.')

        exported_names = ['severity_importance', 'ablation_results', 'combinations', 'rule_model', 'rule_text', 'predict_record_severity', 'scenario_rows', 'daily_importance', 'daily_ablation', 'poisson_coefficients']
        return {name:value for name,value in locals().items() if name in exported_names}

    def module_09_export_results(state):
        """Save fitted models, settings, summary and one ZIP organized by task module.

        Inputs: Completed outputs from all preceding modules.
        Returns: a dictionary of named module outputs."""
        OUTPUT_DIR = OUTPUT_ROOT / '09_exports'
        save_table, finish_figure = output_helpers(OUTPUT_DIR)
        DAILY_FEATURES = state['05_prediction']['DAILY_FEATURES']
        DATE_FIXES = state['01_cleaning']['DATE_FIXES']
        FEATURES = state['04_classification']['FEATURES']
        MC_LABELS = state['04_classification']['MC_LABELS']
        SOURCE_SHA256 = state['00_setup']['SOURCE_SHA256']
        best_classifier = state['04_classification']['best_classifier']
        best_classifier_name = state['04_classification']['best_classifier_name']
        best_forecast_method = state['06_forecasting']['best_forecast_method']
        best_regressor = state['05_prediction']['best_regressor']
        best_regressor_name = state['05_prediction']['best_regressor_name']
        classification_holdout = state['07_performance']['classification_holdout']
        classifier_pipeline = state['04_classification']['classifier_pipeline']
        classifier_specs = state['04_classification']['classifier_specs']
        daily = state['01_cleaning']['daily']
        daily_deployment_model = state['05_prediction']['daily_deployment_model']
        daily_holdout = state['07_performance']['daily_holdout']
        df = state['01_cleaning']['df']
        forecast_holdout = state['07_performance']['forecast_holdout']
        interval_coverage = state['07_performance']['interval_coverage']
        interval_half_width = state['06_forecasting']['interval_half_width']
        mc_name = state['04_classification']['mc_name']
        model_data = state['04_classification']['model_data']
        monthly_series = state['06_forecasting']['monthly_series']
        multiclass_model = state['04_classification']['multiclass_model']
        raw = state['00_setup']['raw']

        severity_deployment_model=classifier_pipeline(classifier_specs[best_classifier_name]).fit(
            model_data[FEATURES],model_data.severe.astype(int))
        joblib.dump({'pipeline':best_classifier,'features':FEATURES,'selected_model':best_classifier_name,
                     'training_period':'2021–2023','target':'major injury or fatal vs minor or no injury'},
                    OUTPUT_DIR/'severity_evaluation_model.joblib')
        joblib.dump({'pipeline':severity_deployment_model,'features':FEATURES,'selected_model':best_classifier_name,
                     'training_period':'reliable injury grades 2021–2024','target':'major injury or fatal vs minor or no injury'},
                    OUTPUT_DIR/'severity_deployment_model.joblib')
        joblib.dump({'pipeline':multiclass_model,'features':FEATURES,'classes':MC_LABELS},
                    OUTPUT_DIR/'multiclass_evaluation_model.joblib')
        joblib.dump({'pipeline':best_regressor,'features':DAILY_FEATURES,'selected_model':best_regressor_name,
                     'forecast_type':'one-day-ahead with observed lag updates','training_period':'through 2024'},
                    OUTPUT_DIR/'daily_count_evaluation_model.joblib')
        joblib.dump({'pipeline':daily_deployment_model,'features':DAILY_FEATURES,'selected_model':best_regressor_name,
                     'last_observed_date':str(daily.index.max().date())},OUTPUT_DIR/'daily_count_deployment_model.joblib')
        joblib.dump({'method':best_forecast_method,'training_series':monthly_series,
                     'approx_interval_half_width':interval_half_width,'forecast_function':'forecast_monthly in this notebook'},
                    OUTPUT_DIR/'monthly_forecast_state.joblib')

        settings={'seed':SEED,'source_sha256':SOURCE_SHA256,'rows':len(raw),'versions':versions,
            'apply_context_date_fixes':APPLY_CONTEXT_DATE_FIXES,'drop_exact_duplicates':DROP_EXACT_DUPLICATES,
            'include_investigated_cause':INCLUDE_INVESTIGATED_CAUSE,'assume_complete_reporting_calendar':ASSUME_COMPLETE_REPORTING_CALENDAR,
            'severity_test_year':SEVERITY_TEST_YEAR,'count_test_year':COUNT_TEST_YEAR,
            'severity_features':FEATURES,'daily_features':DAILY_FEATURES,'forecast_horizon':FORECAST_HORIZON,
            'date_corrections':DATE_FIXES,'selected_binary_classifier':best_classifier_name,
            'selected_multiclass_classifier':mc_name,'selected_daily_regressor':best_regressor_name,
            'selected_monthly_forecaster':best_forecast_method}
        (OUTPUT_DIR/'analysis_settings.json').write_text(json.dumps(settings,indent=2,default=str),encoding='utf-8')
        (OUTPUT_DIR/'requirements_tested.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n',encoding='utf-8')

        selected_classification=classification_holdout.iloc[0]
        selected_daily=daily_holdout.iloc[0]
        selected_forecast=forecast_holdout.iloc[0]
        summary_lines=[
            '# RACC analysis run summary',
            f'Source SHA-256: {SOURCE_SHA256}',
            f'Raw records: {len(raw)}; retained records: {len(df)}; provisional incident keys: {df.incident_proxy.nunique()}.',
            f'Provisional context-dependent dates: {int(df.provisional_date_flag.sum())}; usable injury labels: {int(df.severity_grade.notna().sum())}.',
            '', '## Chronological holdout results',
            f'Binary classifier: {best_classifier_name}; accuracy={selected_classification.accuracy:.3f}; macro-F1={selected_classification.macro_f1:.3f}; ROC-AUC={selected_classification.ROC_AUC:.3f}.',
            f'Daily prediction: {best_regressor_name}; MAE={selected_daily.MAE:.3f}; RMSE={selected_daily.RMSE:.3f}; R2={selected_daily.R2:.3f}.',
            f'Monthly forecasting: {best_forecast_method}; MAE={selected_forecast.MAE:.3f}; RMSE={selected_forecast.RMSE:.3f}; WAPE={selected_forecast.WAPE_pct:.1f}%.',
            f'Approximate 90% monthly interval empirical holdout coverage: {interval_coverage:.1%}.',
            '', '## What the results support',
            ('- Severity discrimination is close to chance. Do not describe ranked features or combinations as validated injury predictors.'
                if selected_classification.ROC_AUC<.55 else
                '- Interpret severity discrimination with the reported holdout metrics, calibration and confidence intervals.'),
            ('- A constant daily-count baseline wins development MAE. Variable importance is shown for a separate learned reference; it does not demonstrate improved prediction over that baseline.'
                if 'baseline' in best_regressor_name.lower() else
                '- Compare daily regression against the mean and previous-week baselines; inspect bias and error metrics together.'),
            ('- The monthly forecaster has negative holdout R2. Any improvement over a naive forecast is relative, not evidence of high absolute forecast accuracy.'
                if selected_forecast.R2<0 else
                '- Monthly forecast performance is specific to this short history and recording process.'),
            '', '## Interpretation limits',
            '- Counts measure reported records. True accident rates require exposure and verified incident IDs.',
            '- Most 2025 severity statuses are consciousness descriptions, not injury grades.',
            '- Date corrections are provisional and require source-owner confirmation.',
            '- Missing calendar dates are assumed to have zero recorded rows; completeness is unverified.',
            '- Feature importance and combinations describe predictions and associations, not causation.',
            '- Dataset-relative extrapolations may include dates already elapsed when this notebook is run.',
            '- Limited data, rare fatal cases and temporal reporting changes limit generalization.',
        ]
        run_summary='\n'.join(summary_lines)
        (OUTPUT_DIR/'RUN_SUMMARY.md').write_text(run_summary,encoding='utf-8')
        display(Markdown(run_summary))

        module_index=pd.DataFrame([
            {'module':key,'task':spec['title'],'prerequisites':', '.join(spec['requires']) or 'none',
             'output_folder':spec['folder']} for key,spec in MODULE_SPECS.items()
        ])
        save_table(module_index,'44_module_directory')
        manifest=pd.DataFrame([{'filename':str(p.relative_to(OUTPUT_ROOT)),'bytes':p.stat().st_size}
            for p in sorted(OUTPUT_ROOT.rglob('*')) if p.is_file() and p.name!='42_output_manifest.csv'])
        save_table(manifest,'42_output_manifest')
        archive=Path(shutil.make_archive(str(OUTPUT_ROOT.resolve()),'zip',root_dir=OUTPUT_ROOT))
        print(f'Created {len(manifest)+1} categorized output files and archive: {archive.name}')

        exported_names = ['severity_deployment_model', 'settings', 'run_summary', 'manifest', 'archive', 'module_index']
        return {name:value for name,value in locals().items() if name in exported_names}

    return {'07_performance': module_07_model_performance, '08_features': module_08_feature_analysis, '09_exports': module_09_export_results}
