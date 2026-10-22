# RACC Streamlit application

Use Python 3.12 (recommended). This package runs locally and includes all analysis functions.
The notebook is not required. No dataset is embedded: upload your own RACC CSV in the app.

## Start

Open a terminal in this extracted folder, then run:

    python -m pip install -r requirements.txt
    python -m streamlit run racc_streamlit_app.py

Streamlit opens the app in your browser. If it does not, open the local URL printed by the terminal.

## Use

1. Upload a CSV in the left panel and click **Load dataset**.
2. Review the cleaning settings. Changing data or settings clears previous results.
3. Click **Run complete analysis**, or choose an analysis task and click **Run this task**.
   Missing prerequisites run automatically. Results persist while you navigate.
4. Inspect **Model performance** before interpreting **Features and combinations**.
5. Use **Downloads** for individual CSVs/PNGs/models or a ZIP of all available outputs.

Tasks: setup/loading; cleaning; statistics; EDA; classification; numeric prediction;
forecasting; performance evaluation; feature combinations; exports.

## CSV format and scope

Required columns (case/outer whitespace are normalized):

    DATE,TIME,LOCATION,WEATHER,ROAD CONDITION,VEHICLE TYPE,CAUSE,SEVERITY,GENDER

This remains a dataset-specific conversion of the RACC notebook, with defaults for 2021–2025.
Other schemas or periods require changes to date parsing, date corrections, targets and backtests.
Core analysis algorithms and default numeric settings are preserved.

Severity models develop on 2021–2023 and evaluate on 2024. Most 2025 severity labels describe
consciousness and are excluded from injury-grade classification. Daily count prediction and
monthly forecasting evaluate on 2025, with different prediction horizons and protocols.

Exact repeated rows are retained by default; provisional date repairs are reviewable.
Missing calendar days are assumed to have zero reported rows, not zero true accidents.
Counts do not establish accident rates. Feature importance and combinations are not causal effects.
Weak predictive performance is reported as observed. Forecasts follow the dataset end, which may
already be in the past when you run the app. Intervals are approximate.

Each browser session has an isolated temporary output directory. Restarting the server may
discard session results; download them first. Fitted joblib models require a compatible Python
environment and should only be loaded from trusted sources.

## Technical references

- Streamlit uploader: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- Streamlit session state: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
- Model evaluation and pipelines: https://scikit-learn.org/stable/common_pitfalls.html
