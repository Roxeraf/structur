import streamlit as st
import pandas as pd
import os
from io import BytesIO
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, DataCleaningTool
from langchain.llms import OpenAI

# Setzen Sie Ihren OpenAI API-Schlüssel
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Initialisieren Sie das GPT-4o-mini Modell
llm = OpenAI(model_name="gpt-4o-mini")

# Tools initialisieren
search_tool = SerperDevTool()
cleaning_tool = DataCleaningTool()

# Agenten definieren
data_researcher = Agent(
    role='Datenforscher',
    goal='Führe eine detaillierte Analyse der Datenstrukturen durch und identifiziere Muster.',
    verbose=True,
    memory=True,
    backstory='Du bist ein erfahrener Datenwissenschaftler mit einem starken Fokus auf Datenmuster und Analysen.',
    tools=[search_tool]
)

data_cleaner = Agent(
    role='Datenbereinigungsexperte',
    goal='Bereinige und bereite die Daten für die Analyse vor.',
    verbose=True,
    memory=True,
    backstory='Du bist ein Experte für Datenbereinigung und Vorbereitung, um sicherzustellen, dass die Daten für die Analyse geeignet sind.',
    tools=[cleaning_tool]
)

reporter = Agent(
    role='Berichterstatter',
    goal='Erstelle einen umfassenden Bericht basierend auf den Analysen.',
    verbose=True,
    memory=True,
    backstory='Du bist ein erfahrener Berichterstatter, der komplexe Datenanalysen in verständliche Berichte umwandelt.',
    tools=[llm]
)

# Tasks definieren
analysis_task = Task(
    description='Analysiere die Datenstrukturen und identifiziere relevante Muster. Dein Bericht sollte die wichtigsten Punkte klar artikulieren.',
    expected_output='Ein detaillierter Bericht über die Datenstrukturanalyse.',
    tools=[search_tool],
    agent=data_researcher
)

cleaning_task = Task(
    description='Bereinige die Daten und bereite sie für die Analyse vor.',
    expected_output='Bereinigte und vorbereitete Daten.',
    tools=[cleaning_tool],
    agent=data_cleaner
)

reporting_task = Task(
    description='Erstelle einen umfassenden Bericht basierend auf den Ergebnissen der Analyse.',
    expected_output='Ein umfassender Bericht basierend auf den Analyseergebnissen.',
    agent=reporter,
    async_execution=False,
    output_file='data-analysis-report.md'
)

# Crew erstellen
crew = Crew(
    agents=[data_researcher, data_cleaner, reporter],
    tasks=[analysis_task, cleaning_task, reporting_task],
    process=Process.sequential
)

# Streamlit App
st.title("Datenstrukturanalyse App")
st.write("Laden Sie eine CSV- oder Excel-Datei hoch, um die Analyse zu starten.")

uploaded_file = st.file_uploader("Wählen Sie eine Datei aus", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.type == "text/csv":
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write("Hochgeladene Datei:")
    st.dataframe(data)
    
    # Daten in BytesIO speichern für die Crew
    data_io = BytesIO()
    if uploaded_file.type == "text/csv":
        data.to_csv(data_io, index=False)
    else:
        data.to_excel(data_io, index=False)
    data_io.seek(0)
    
    # Crew Kickoff
    result = crew.kickoff(inputs={'data': data_io})
    
    st.write("Analyseergebnis:")
    st.write(result)
