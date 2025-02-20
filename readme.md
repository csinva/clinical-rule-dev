<h1>Analyzing patient perspectives with LLMs </h1>
<p>
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.9--3.11-blue">
  <img src="https://img.shields.io/badge/numpy-<2.0-darkgreen">
</p>

Code for auditing patient perspectives, underlying a cross-sectional study of sentiment and thematic classification on exception from informed consent (<a href="https://www.nature.com/articles/s41598-025-89996-w">Kornblith et al. 2025, Nature scientific reports</a>) <br/>
<img align="center" width=100% src="https://csinva.io/analyzing-patient-psrspectives/figs/sentiment_agg.png"> </img>


### Abstract

> Large language models (LLMs) can improve text analysis efficiency in healthcare. This study explores the application of LLMs to analyze patient perspectives within the exception from informed consent (EFIC) process, which waives consent in emergency research. Our objective is to assess whether LLMs can analyze patient perspectives in EFIC interviews with performance comparable to human reviewers. We analyzed 102 EFIC community interviews from 9 sites, each with 46 questions, as part of the Pediatric Dose Optimization for Seizures in Emergency Medical Services study. We evaluated 5 LLMs, including GPT-4, to assess sentiment polarity on a 5-point scale and classify responses into predefined thematic classes. Three human reviewers conducted parallel analyses, with agreement measured by Cohen’s Kappa and classification accuracy. Polarity scores between LLM and human reviewers showed substantial agreement (Cohen’s kappa: 0.69, 95% CI 0.61–0.76), with major discrepancies in only 4.7% of responses. LLM achieved high thematic classification accuracy (0.868, 95% CI 0.853–0.881), comparable to inter-rater agreement among human reviewers (0.867, 95% CI 0.836–0.901). LLMs enabled large-scale visual analysis, comparing response statistics across sites, questions, and classes. LLMs efficiently analyzed patient perspectives in EFIC interviews, showing substantial sentiment assessment and thematic classification performance. However, occasional underperformance suggests LLMs should complement, not replace, human judgment. Future work should evaluate LLM integration in EFIC to enhance efficiency, reduce subjectivity, and support accurate patient perspective analysis.