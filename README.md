# SkillCorner X PySport Analytics Cup
This repository contains the submission template for the SkillCorner X PySport Analytics Cup **Analyst Track**. 
Your submission for the **Analyst Track** should be on the `main` branch of your own fork of this repository.

Find the Analytics Cup [**dataset**](https://github.com/SkillCorner/opendata/tree/master/data) and [**tutorials**](https://github.com/SkillCorner/opendata/tree/master/resources) on the [**SkillCorner Open Data Repository**](https://github.com/SkillCorner/opendata).

## Submitting
Make sure your `main` branch contains:

1. A single Jupyter Notebook in the root of this repository called `submission.ipynb`
    - This Juypter Notebook can not contain more than 2000 words.
    - All other code should also be contained in this repository, but should be imported into the notebook from the `src` folder.


or,


1. A single Python file in the root of this repository called `main.py`
    - This file should not contain more than 2000 words.
    - All other code should also be contained in this repository, but should be imported into the notebook from the `src` folder.

or, 


1. A publicly accessible web app or website written in a language of your choice (e.g. Javascript)

    - Your code should follow a clear and well defined structure.
    - All other code should also be contained in this repository.
    - The URL to the webapp should be included at the bottom of the read me under **URL to Web App / Website**


2. An abstract of maximum 300 words that follows the **Analyst Track Abstract Template**.
3. Add a URL to a screen recording video of maximum 60 seconds that shows your work. Add it under the **Video URL** Section below. (Use YouTube, or any other site to share this video).
4. Submit your GitHub repository on the [Analytics Cup Pretalx page](https://pretalx.pysport.org)

Finally:
- Make sure your GitHub repository does **not** contain big data files. The tracking data should be loaded directly from the [Analytics Cup Data GitHub Repository](https://github.com/SkillCorner/opendata). For more information on how to load the data directly from GitHub please see this [Jupyter Notebook](https://github.com/SkillCorner/opendata/blob/master/resources/getting-started-skc-tracking-kloppy.ipynb).
- Make sure the `submission.ipynb` notebook runs on a clean environment, or
- Provide clear and concise instructions how to run the `main.py` (e.g. `streamlit run main.py`) if applicable in the **Run Instructions** Section below.
- Providing a URL to a publically accessible webapp or website with a running version of your submission is mandatory when choosing to submit in a different language then Python, it is encouraged, but optional when submitting in Python.

_⚠️ Not adhering to these submission rules and the [**Analytics Cup Rules**](https://pysport.org/analytics-cup/rules) may result in a point deduction or disqualification._

---

## Analyst Track Abstract Template (max. 300 words)
#### Introduction

The Phase of Play Transition Analyzer is a comprehensive analytical tool designed to help football coaches and analysts understand tactical patterns through the analysis of phase transitions in matches. By leveraging SkillCorner's phases of play data, the tool identifies how teams move through different tactical phases (create, build-up, finish, transition, etc.) and quantifies the effectiveness of these transitions in terms of goals and shots generated.

The tool processes phases of play data directly from the SkillCorner Open Data repository, identifying transitions between phases within possessions, calculating success rates, and visualizing patterns through network diagrams, spatial heatmaps, and sequence analysis. This enables a data-driven understanding of tactical behaviors that goes beyond traditional event-based analysis.

#### Usecase(s)

1. **Tactical Analysis**: Coaches can identify which phase transitions are most effective for their team's playing style, helping them optimize their tactical approach and training focus.

2. **Opposition Analysis**: Analysts can study opponent transition patterns to develop defensive strategies, identify vulnerabilities, and prepare match-specific game plans.

3. **Player Development**: The tool helps coaches train players to recognize and execute successful transition sequences, improving tactical awareness and decision-making.

4. **Performance Evaluation**: Teams can compare their transition patterns across matches, periods, and against different opponents to track tactical evolution and effectiveness.

5. **Spatial Insights**: Understanding where on the pitch successful transitions occur helps teams design training sessions and tactical patterns that maximize goal-scoring opportunities.

#### Potential Audience

- **Football Coaches**: First-team coaches, academy coaches, and tactical analysts seeking to understand and improve team performance through phase transition analysis.

- **Sports Analysts**: Professional analysts working with clubs, media, or betting companies who need to quantify and visualize tactical patterns.

- **Performance Analysts**: Data scientists and analysts working in football who want to apply advanced analytics to phases of play data.

- **Researchers**: Academics and researchers studying football tactics and team performance who need tools to analyze large datasets of match data.

---

## Video URL

[Video URL to be added]

---

## Run Instructions

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd analytics_cup_analyst
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Jupyter Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `submission.ipynb` and run all cells to see the analysis.

### Running the Streamlit Web App

1. Run the Streamlit application:
```bash
streamlit run main.py
```

2. The app will open in your default web browser at `http://localhost:8501`

3. Use the sidebar to:
   - Select matches to analyze
   - Apply filters (phase types, periods, outcomes, teams)
   - Navigate between different analysis views

### Data Loading

The tool loads data directly from the SkillCorner Open Data GitHub repository. No local data files are required. The first run may take a few moments to download the data.

---

## [Optional] URL to Web App / Website

[Web App URL to be added if deployed]