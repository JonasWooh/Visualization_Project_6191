# Visualization_Project_6191
Repository for my course FRE 6191 project

Exploratory data analysis (EDA) and visualization around **YouTube Shorts** and **TikTok**.
The repo includes Jupyter notebooks, an interactive dashboard (code/app.py), and exported figures.

## Features
- Data cleaning and metric building (engagement rate, watch time, creator tiers, etc.)
- Interactive dashboard for platform comparisons, time-based analysis, and topic/hashtag distributions
- Reproducible chart exports and scripts

## Project Structure
```text
.
├─ code/
│  ├─ Shorts_TikTok_Trends_EDA.ipynb
│  ├─ Viz_Project.ipynb
│  ├─ YouTube_TikTok_Interactive_Dash.ipynb
│  └─ app.py  # Dashboard entry (Plotly Dash)
├─ assets/
│  └─ style.css
├─ data/
│  ├─ USvideos.csv
│  └─ youtube_shorts_tiktok_trends_2025.csv
├─ result_pic/   # Exported charts and report assets
├─ environment.yml  # Conda environment specification
└─ README.md
```

## Quick Start

1) Environment
   - Create and activate the Conda environment defined in environment.yml
     conda env create -f environment.yml
     conda activate <environment-name>   # replace with the "name" field in environment.yml

2) Notebooks
   - Launch Jupyter
     jupyter lab   (or: jupyter notebook)

3) Run the Dashboard (Plotly Dash)
   - From the repo root
     python code/app.py
   - The app will start locally; check the console for the URL (typically http://127.0.0.1:8050)

## Data
- Sample data is placed under the "data/" directory. Replace or extend with your own sources as needed and document provenance.
- For large files, consider Git LFS to avoid GitHub’s 100 MB hard limit.

## Results
- Key figures and exported images live in "result_pic/".

## Development Notes
- Keep temporary files, caches, and secrets out of version control (e.g., __pycache__/, .ipynb_checkpoints/, .env).
- If collaborating across OSes, consider normalizing line endings via a repository-level .gitattributes (e.g., "* text=auto eol=lf").
- Large binaries can be tracked with Git LFS:
     git lfs install
     git lfs track "result_pic/**.png" "*.pdf"
     git add .gitattributes
     git commit -m "chore: track large binaries via Git LFS"

## License
Not specified yet. If you plan to open source, consider MIT or Apache-2.0.

## Acknowledgments
Thanks to open data sources and the Python ecosystem (Pandas, Plotly/Matplotlib, Jupyter, etc.).

