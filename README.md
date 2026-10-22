# RACC app in four Python parts

Use Python 3.12. Keep these four Python files together in the same folder:

| Part | File | Responsibility |
|---|---|---|
| 1 | racc_streamlit_app.py | Streamlit entry point, upload, shared state and four-stage loading |
| 2 | part_2_data.py | Data cleaning, statistics and exploratory analysis |
| 3 | part_3_models.py | Classification, daily count prediction and monthly forecasting |
| 4 | part_4_results.py | Model performance, feature combinations and exports |

Do not paste all four files into one editor pane. Create each named file separately.
The filenames must match exactly. Only run the Part 1 entry file; it imports the other files
when their tasks are requested. The three supporting files are not standalone Streamlit apps.

## Local run

Extract the ZIP, open a terminal in the extracted folder, and run:

    python -m pip install -r requirements.txt
    python -m streamlit run racc_streamlit_app.py

## Run in four stages

1. In **Part 1**, upload your CSV and click **Load dataset — Part 1**.
2. Select **Part 2 — Data analysis**, then **Run Part 2**. Review cleaning, statistics and EDA.
3. Select **Part 3 — Models and forecasting**, then **Run Part 3**.
4. Select **Part 4 — Evaluation and downloads**, then **Run Part 4**. Review performance,
   variable combinations and download results. Each part keeps previous results in the session.

The app no longer starts all analysis tasks with one button. Changing the dataset or settings
clears the old results; start again at Part 1. The split reduces source-file size and stages
the work. It does not increase the hosting service's RAM or bypass its resource limits.

## Streamlit Community Cloud

Place all four Python files and requirements.txt in the same repository folder. Choose
**racc_streamlit_app.py** as the app entry point. Do not choose part_2_data.py, part_3_models.py
or part_4_results.py as the entry point. Use Python 3.12 in deployment settings.

Official file organization:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization

## Dataset and interpretation

Required CSV headers:

    DATE,TIME,LOCATION,WEATHER,ROAD CONDITION,VEHICLE TYPE,CAUSE,SEVERITY,GENDER

This remains the RACC-specific workflow with January 2021–December 2025 defaults. Another
schema or recording period needs changes to date parsing, corrections and evaluation windows.
The original algorithms and default parameters are preserved. Consciousness labels are not
injury grades. Repeated records and incomplete reporting need review. Counts are not exposure-
adjusted accident rates, and variable importance does not establish causation. Forecast dates
follow the dataset end and may already be in the past. Download results before restarting the
server, since session output files are temporary.
