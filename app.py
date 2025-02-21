import streamlit as st
from app_pages.multi_page import MultiPage
from app_pages import page_proj_summary, page_leaves_visualizer, page_mildew_detector, page_ml_performance, page_proj_hypothesis

app = MultiPage("Cherry Leaf Mildew Detection")

# Pages
app.add_page("Project Summary", page_proj_summary.run)
app.add_page("Leaves Visualizer", page_leaves_visualizer.run)
app.add_page("Mildew Detector", page_mildew_detector.run)
app.add_page("ML Performance", page_ml_performance.run)
app.add_page("Project Hypothesis", page_proj_hypothesis.run)

# Run the app
app.run()