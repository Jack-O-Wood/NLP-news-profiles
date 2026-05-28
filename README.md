# Multi-Dimensional News Outlet Coverage Profiles

A multi-method NLP analysis of how five news outlets (CNN, BBC, Associated Press, Fox News, and NewsMax) frame coverage across five contested topics. The paper argues that single-axis bias ratings collapse useful information about *how* outlets cover news, and builds a multi-dimensional profile approach that surfaces patterns those ratings cannot capture.

**Paper:** [News Profiles Paper.pdf](./News_Profiles_Paper_Jack_Wood.pdf)

**Interactive explorer:** [Hugging Face Space](https://huggingface.co/spaces/Jack-O-Wood/news-outlet-profiles)

## Methods

The analysis combines six complementary methods, applied to article bodies and headlines separately throughout:

- **Lexical signatures** via TF-IDF
- **Sentiment** via VADER (with a headline / body divergence measure)
- **Discrete emotion** via the NRC Emotion Lexicon (eight emotions plus a composite "outrage" measure)
- **Moral framing** via the Moral Foundations Dictionary (Care/Harm, Fairness, Loyalty, Authority, Purity)
- **Sub-theme extraction** via per-topic non-negative matrix factorization (NMF)
- **Named-entity recognition** via spaCy

A three-layer topic-conditioned blocklist removes keyword-related vocabulary before lexical analysis, addressing a problem specific to keyword-sampled corpora where topic terms otherwise dominate TF-IDF and NMF results.

## Corpus

- 2,334 full-text articles
- 5 outlets: CNN, BBC, Associated Press, Fox News, NewsMax
- 5 topics: AI, Climate, Economy, Immigration, Iran
- Date range: January 1 to April 22, 2026
- Sourced via the MediaCloud API; full text extracted with trafilatura

## Reproducing the analysis

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download spaCy and NLTK assets:
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
   ```

3. For data collection, set the MediaCloud API token as an environment variable:
   ```bash
   export MEDIACLOUD_API_TOKEN=<your-token>
   ```

4. To rebuild the corpus from scratch, run `notebooks/mediacloud_collect_data.ipynb`. Otherwise, the cleaned corpus is already available at `data/mediacloud_articles_clean.csv`.

5. Run `notebooks/news_profiles_analysis.ipynb` to regenerate all metrics, figures, and exported CSVs in `profile_data/`.

6. Launch the interactive explorer locally:
   ```bash
   python app.py
   ```

## Repository structure

```
news-outlet-profiles/
├── README.md
├── LICENSE
├── requirements.txt
├── app.py                              (Gradio explorer)
├── paper/
│   └── News Profiles Paper.pdf
├── notebooks/
│   ├── mediacloud_collect_data.ipynb   (data collection)
│   └── news_profiles_analysis.ipynb    (main analysis)
├── data/
│   └── mediacloud_articles_clean.csv   (final cleaned corpus)
└── profile_data/                       (precomputed outputs that feed app.py)
    ├── sentiment_body.csv
    ├── sentiment_headline.csv
    ├── nrc_body_by_outlet.csv
    ├── nrc_headline_by_outlet.csv
    ├── mft_by_outlet.csv
    ├── mft_by_topic_outlet.csv
    ├── entities.csv
    ├── tfidf_top_terms_by_outlet.csv
    └── ... (additional per-topic and per-outlet breakdowns)
```

## Key findings

The full discussion is in the paper. Briefly:

- Fox News intensifies fear-coded vocabulary in headlines; CNN leads the corpus on anger-coded vocabulary.
- NewsMax is an outlier on fairness-coded moral language, scoring more than double the corpus mean.
- BBC sits structurally outside the U.S. political register, with the lowest moral-language density in the corpus and a near-absent American national identity in named entities.
- Outlets across the political spectrum foreground entities representing the opposite of their stated ideological lean.
- A case study on immigration coverage shows the per-outlet patterns sharpening rather than dissolving under topic constraint.

## Citation

```
Wood, J. (2026). Multi-Dimensional Coverage Profiles: A Lexical,
Affective, and Moral Comparison of How Five U.S. News Outlets
Cover Five Contested Topics. Fordham University, Natural Language Processing, 2026.
```

## Contact

wood.jack26@gmail.com, https://www.linkedin.com/in/jack-oliver-wood-baa734114/
