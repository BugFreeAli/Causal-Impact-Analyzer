📈 Causal Impact Analyzer

An interactive Streamlit application for estimating and visualizing the causal effect of interventions on time series data using Bayesian Structural Time Series (BSTS) models.

This tool is designed for data scientists, analysts, and researchers who want to measure the true impact of events (e.g., product launches, marketing campaigns, policy changes) on KPIs in a professional and intuitive way.

🚀 Features

📁 Easy Data Upload
 Upload CSV files with date and metric columns.

⚙️ Smart Preprocessing
 Handles duplicate dates automatically
 Detects and fills missing days (forward fill / linear interpolation)
 Generates a clean, complete time series

📊 Data Exploration
 Interactive preview of uploaded data
 Data quality summary (mean, std dev, min, max, coverage)
 Trend visualization with event markers

🔍 Causal Impact Analysis
 Bayesian structural time series (BSTS) model via causalimpact
 Customizable seasonality & advanced model args
 Automatic detection of actual vs predicted values

Key metrics: significance, p-values, effect sizes

📈 Interactive Visualization
Plotly-based plots:
 Actual vs Predicted with confidence intervals
 Pointwise causal effect over time
 Red vertical marker for intervention date

📤 Export Results
 Download summary report as text
 Export interactive plots as PNG (via kaleido)
 Export processed results & inferences as CSV

🛠️ Tech Stack
Python 3.9+
Streamlit → UI framework
Pandas / NumPy → Data processing
CausalImpact → Bayesian Structural Time Series modeling
Plotly → Interactive visualizations
Kaleido (optional) → Export charts as PNG

📷 Screenshots

![Data Overview](assets/overview.png)

Causal Impact Analysis

![Data Overview](assets/impact.png)

📦 Installation

Clone the repo:

git clone https://github.com/your-username/causal-impact-analyzer.git
cd causal-impact-analyzer


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)


Install dependencies:

pip install -r requirements.txt

▶️ Run the App
streamlit run app.py


The app will launch in your browser at http://localhost:8501
.

📂 Example Data

Example dataset format:

Date	Sales
2023-01-01	100
2023-01-02	103
...	...

Date → Date column (daily frequency recommended)

Metric (e.g., Sales) → Numeric values to analyze

🌟 Why This Project?

This project demonstrates my ability to:

Build end-to-end data apps with professional UI/UX in Streamlit

Apply advanced causal inference methods (BSTS, Bayesian analysis)

Handle real-world data cleaning challenges (missing dates, duplicates, gaps)

Integrate interactive visualizations with Plotly

Provide downloadable insights (reports, charts, raw results) for stakeholders

📌 Future Improvements

Multi-metric causal impact (support covariates)

Automatic report generation (PDF)

Deployment to Streamlit Cloud / Docker

Forecasting future impact scenarios

👨‍💻 Author

Ali Akbar Gondal
📌 Computer Science Student | Data Scientist | Developer

✨ If you like this project, give it a ⭐ on GitHub to support!
