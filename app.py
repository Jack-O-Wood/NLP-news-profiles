import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import json

# ── Load data ──────────────────────────────────────────────────────────────────
DATA = "profile_data"

counts        = pd.read_csv(f"{DATA}/article_counts.csv")
sent_body     = pd.read_csv(f"{DATA}/sentiment_body.csv")
sent_headline = pd.read_csv(f"{DATA}/sentiment_headline.csv")
divergence    = pd.read_csv(f"{DATA}/divergence_by_outlet.csv")
nrc_body      = pd.read_csv(f"{DATA}/nrc_body_by_outlet.csv")
nrc_headline  = pd.read_csv(f"{DATA}/nrc_headline_by_outlet.csv")
nrc_body_t    = pd.read_csv(f"{DATA}/nrc_body_by_outlet_topic.csv")
nrc_head_t    = pd.read_csv(f"{DATA}/nrc_headline_by_outlet_topic.csv")
sent_topic    = pd.read_csv(f"{DATA}/sentiment_by_outlet_topic.csv")
outrage       = pd.read_csv(f"{DATA}/outrage_gap_by_outlet.csv")
mft           = pd.read_csv(f"{DATA}/mft_by_outlet.csv")
mft_topic     = pd.read_csv(f"{DATA}/mft_by_topic_outlet.csv")
ent_df        = pd.read_csv(f"{DATA}/entities.csv")
tfidf         = pd.read_csv(f"{DATA}/tfidf_top_terms_by_outlet.csv")

with open(f"{DATA}/nmf_custom_labels.json") as f:
    nmf_labels = json.load(f)

nmf_props      = pd.read_csv(f"{DATA}/nmf_outlet_topic_proportions.csv")
headlines_df   = pd.read_csv(f"{DATA}/headlines_outrage.csv")

OUTLETS = sorted(counts["outlet"].unique())
TOPICS  = ["All"] + sorted(ent_df["topic"].dropna().unique())
EMOTIONS = ["anger", "fear", "disgust", "sadness", "joy", "anticipation", "surprise", "trust"]
MFT_DIMS = ["care", "fair", "ingroup", "order", "purity"]
MFT_LABELS = {"care": "Care / Harm", "fair": "Fairness", "ingroup": "Loyalty", "order": "Authority", "purity": "Purity"}
ENTITY_TYPES = ["PERSON", "ORG", "GPE", "NORP"]

OUTLET_COLORS = {
    "AP":       "#636EFA",
    "BBC":      "#EF553B",
    "CNN":      "#00CC96",
    "Fox News": "#AB63FA",
    "Newsmax":  "#FFA15A",
}

RADAR_MAX = 8.5

CSS = """
body, .gradio-container, .markdown-text, .prose, p, li, span, label {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
h1, h2, h3, h4 { font-family: 'Inter', system-ui, sans-serif !important; }
.entity-section h4 { margin: 12px 0 4px 0; font-size: 13px; text-transform: uppercase;
                      letter-spacing: 0.08em; color: #666; }
"""

# ── Helpers ────────────────────────────────────────────────────────────────────
def _row(df, **filters):
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in filters.items():
        mask &= df[col] == val
    rows = df[mask]
    return rows.iloc[0] if len(rows) else None


def rank_label(series, outlet, ascending=True):
    ranked = series.rank(ascending=ascending, method="min").astype(int)
    r = int(ranked[series.index == outlet].values[0])
    return f"#{r} of {len(series)}"


# ── Prose summary (returns HTML) ───────────────────────────────────────────────
OUTRAGE_EMOTIONS = ["anger", "fear", "disgust"]


def build_summary(outlet, topic):
    c     = _row(counts, outlet=outlet)
    color = OUTLET_COLORS[outlet]

    # Sentiment (body, headline, divergence) — topic-aware
    if topic == "All":
        sb = _row(sent_body, outlet=outlet)
        sh = _row(sent_headline, outlet=outlet)
        body_cpd = sb["compound"]
        head_cpd = sh["compound"]
        div_val  = _row(divergence, outlet=outlet)["mean_divergence"]
    else:
        st = _row(sent_topic, outlet=outlet, topic=topic)
        body_cpd = st["body_compound"]     if st is not None else 0
        head_cpd = st["headline_compound"] if st is not None else 0
        # Topic-specific divergence approximated as difference of means
        div_val  = head_cpd - body_cpd

    def tone(v):
        return "positive" if v > 0.05 else ("negative" if v < -0.05 else "neutral")

    # Outrage gap and rank — topic-aware
    if topic == "All":
        og_val    = float(_row(outrage, outlet=outlet)["outrage_gap"])
        og_series = outrage.set_index("outlet")["outrage_gap"]
    else:
        body_t = nrc_body_t[nrc_body_t["topic"] == topic].set_index("outlet")
        head_t = nrc_head_t[nrc_head_t["topic"] == topic].set_index("outlet")
        og_series = (head_t[OUTRAGE_EMOTIONS].sum(axis=1)
                     - body_t[OUTRAGE_EMOTIONS].sum(axis=1)).round(2)
        og_val = float(og_series.loc[outlet]) if outlet in og_series.index else 0.0

    og_rank = rank_label(og_series, outlet, ascending=False)

    # MFT — topic-aware (mft.csv is pre-×1000; mft_topic.csv stores raw ratios)
    if topic == "All":
        mf = _row(mft, outlet=outlet)
        mft_scale = 1
    else:
        mf = _row(mft_topic, outlet=outlet, topic=topic)
        mft_scale = 1000

    dominant_mft = max(MFT_DIMS, key=lambda d: mf[d])
    dominant_val = float(mf[dominant_mft]) * mft_scale
    mft_descriptions = {
        "care":    "protection, harm, and welfare",
        "fair":    "justice, rights, and reciprocity",
        "ingroup": "loyalty, patriotism, and group identity",
        "order":   "authority, tradition, and institutional deference",
        "purity":  "sanctity, disgust, and moral cleanliness",
    }
    topic_note = f'<span style="font-size:13px;color:#888;font-weight:400"> — {topic} topic</span>' if topic != "All" else ""

    def stat_card(value, label, highlight=False):
        val_color = color if highlight else "#1a1a2e"
        return f"""
        <div style="background:#f8f9fc;border-left:4px solid {color};border-radius:6px;
                    padding:14px 18px;min-width:130px;flex:1">
            <div style="font-size:26px;font-weight:700;color:{val_color};line-height:1.1">{value}</div>
            <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.07em;margin-top:4px">{label}</div>
        </div>"""

    sent_color = "#2ecc71" if body_cpd > 0.05 else ("#e74c3c" if body_cpd < -0.05 else "#888")

    html = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;padding:4px 0 16px 0">
        <h2 style="font-size:32px;font-weight:800;color:{color};margin:0 0 4px 0;letter-spacing:-0.5px">
            {outlet}{topic_note}
        </h2>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin:16px 0">
            {stat_card(int(c['n_articles']), "Articles")}
            {stat_card(f"~{int(c['avg_length'])} words", "Avg article length")}
            {stat_card(f'<span style="color:{sent_color}">{body_cpd:+.3f}</span>', "Body sentiment")}
            {stat_card(f"{og_val:.2f}", "Headline outrage gap", highlight=True)}
            {stat_card(og_rank, "Outrage rank")}
        </div>
        <p style="font-size:14px;color:#333;line-height:1.65;margin:8px 0 4px 0">
            <strong>Tone:</strong> Body text is {tone(body_cpd)} (VADER {body_cpd:+.3f}); headlines are {tone(head_cpd)} ({head_cpd:+.3f}).
            Headlines tend to run {'more negative' if div_val < 0 else 'more positive'} than the body
            (divergence {div_val:+.3f}){', a pattern consistent with sensationalist framing.' if abs(div_val) > 0.05 else '.'}
        </p>
        <p style="font-size:14px;color:#333;line-height:1.65;margin:4px 0">
            <strong>Moral framing:</strong> Strongest signal in <strong>{MFT_LABELS[dominant_mft]}</strong>
            ({dominant_val:.2f} per 1,000 words), language emphasising {mft_descriptions[dominant_mft]}.
        </p>
    </div>"""
    return html


# ── NRC radar chart ────────────────────────────────────────────────────────────
def nrc_radar(outlet, topic):
    color = OUTLET_COLORS[outlet]

    if topic == "All":
        body_row = _row(nrc_body, outlet=outlet)
        head_row = _row(nrc_headline, outlet=outlet)
    else:
        body_row = _row(nrc_body_t, outlet=outlet, topic=topic)
        head_row = _row(nrc_head_t, outlet=outlet, topic=topic)

    fig = go.Figure()
    for label, row, dash in [("Body", body_row, "solid"), ("Headline", head_row, "dash")]:
        if row is None:
            continue
        vals = [float(row[e]) for e in EMOTIONS] + [float(row[EMOTIONS[0]])]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=EMOTIONS + [EMOTIONS[0]],
            mode="lines+markers", name=label,
            line=dict(color=color, dash=dash, width=2.5),
            marker=dict(size=5),
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, RADAR_MAX], tickfont=dict(size=10))),
        title=dict(text="Emotion Density (per 100 words)", x=0.5, font=dict(size=13)),
        legend=dict(orientation="h", x=0.3, y=-0.12, font=dict(size=12)),
        margin=dict(t=60, b=50, l=60, r=60),
        height=400,
    )
    return fig


# ── MFT bar chart ──────────────────────────────────────────────────────────────
def mft_bar(outlet, topic):
    color = OUTLET_COLORS[outlet]

    if topic == "All":
        row = _row(mft, outlet=outlet)
    else:
        row = _row(mft_topic, outlet=outlet, topic=topic)

    if row is None:
        return go.Figure()

    labels = [MFT_LABELS[d] for d in MFT_DIMS]
    scale  = 1000 if topic != "All" else 1  # topic CSV stores raw ratios, outlet CSV is pre-×1000
    values = [float(row[d]) * scale for d in MFT_DIMS]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=color, opacity=0.85,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Moral Foundations (per 1,000 words)", x=0.5, font=dict(size=13)),
        yaxis=dict(title="Score", range=[0, max(values) * 1.25]),
        xaxis=dict(tickfont=dict(size=12)),
        height=340,
        margin=dict(t=60, b=40, l=50, r=20),
    )
    return fig


# ── Single-outlet entity bar chart ────────────────────────────────────────────
def entity_chart(outlet, etype, topic, n=15):
    sub = ent_df[ent_df["outlet"] == outlet]
    if topic != "All":
        sub = sub[sub["topic"] == topic]
    sub = sub[sub["entity_type"] == etype]
    counts = sub["entity"].value_counts().head(n).reset_index()
    counts.columns = ["entity", "count"]
    counts = counts.sort_values("count")  # ascending for horizontal bar

    color = OUTLET_COLORS[outlet]
    fig = go.Figure(go.Bar(
        x=counts["count"], y=counts["entity"],
        orientation="h",
        marker_color=color, opacity=0.85,
    ))
    fig.update_layout(
        title=dict(text=f"Top {n} {etype} Mentions", x=0.5, font=dict(size=13)),
        xaxis=dict(title="Mentions"),
        yaxis=dict(automargin=True),
        height=420,
        margin=dict(l=160, t=60, b=40, r=20),
    )
    return fig


# ── Signature vocabulary ───────────────────────────────────────────────────────
def signature_terms_md(outlet, n=15):
    top = tfidf[tfidf["outlet"] == outlet].head(n)["term"].tolist()
    return "#### Signature Vocabulary (TF-IDF)\n\n" + "  ·  ".join(top)


# ── Sentiment by topic bar ─────────────────────────────────────────────────────
def sentiment_by_topic_chart(outlet):
    sub = sent_topic[sent_topic["outlet"] == outlet].copy()
    sub = sub.sort_values("topic")
    color = OUTLET_COLORS[outlet]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Body", x=sub["topic"], y=sub["body_compound"],
        marker_color=color, opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        name="Headline", x=sub["topic"], y=sub["headline_compound"],
        marker_color=color, opacity=0.45,
    ))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=dict(text="Sentiment by Topic (VADER compound)", x=0.5, font=dict(size=13)),
        barmode="group",
        yaxis=dict(title="Compound score", zeroline=False),
        legend=dict(orientation="h", x=0.35, y=1.12),
        height=340,
        margin=dict(t=60, b=40, l=50, r=20),
    )
    return fig


# ── Outrage by topic bar ───────────────────────────────────────────────────────
def outrage_by_topic_chart(outlet):
    OUTRAGE_EMOTIONS = ["anger", "fear", "disgust"]
    color = OUTLET_COLORS[outlet]

    body_sub = nrc_body_t[nrc_body_t["outlet"] == outlet].copy()
    head_sub = nrc_head_t[nrc_head_t["outlet"] == outlet].copy()

    body_sub["outrage"] = body_sub[OUTRAGE_EMOTIONS].sum(axis=1)
    head_sub["outrage"] = head_sub[OUTRAGE_EMOTIONS].sum(axis=1)

    merged = body_sub[["topic", "outrage"]].merge(
        head_sub[["topic", "outrage"]], on="topic", suffixes=("_body", "_headline")
    ).sort_values("topic")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Body", x=merged["topic"], y=merged["outrage_body"],
        marker_color=color, opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        name="Headline", x=merged["topic"], y=merged["outrage_headline"],
        marker_color=color, opacity=0.45,
    ))
    fig.update_layout(
        title=dict(text="Outrage Density by Topic (per 100 words)", x=0.5, font=dict(size=13)),
        barmode="group",
        yaxis=dict(title="Outrage density"),
        legend=dict(orientation="h", x=0.35, y=1.12),
        height=340,
        margin=dict(t=60, b=40, l=50, r=20),
    )
    return fig


# ── Profile builder ────────────────────────────────────────────────────────────
def generate_profile(outlet, topic):
    return (
        build_summary(outlet, topic),
        nrc_radar(outlet, topic),
        mft_bar(outlet, topic),
        signature_terms_md(outlet),
    )

def generate_entity_tab(outlet, etype, topic):
    return entity_chart(outlet, etype, topic)


# ── Top outrageous headlines ───────────────────────────────────────────────────
def top_headlines_html(outlet, topic, n=5):
    sub = headlines_df[headlines_df["outlet"] == outlet]
    if topic != "All":
        sub = sub[sub["topic"] == topic]
    top = sub.nlargest(n, "headline_outrage_dens")

    color = OUTLET_COLORS[outlet]
    items = ""
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        score = round(float(row["headline_outrage_dens"]) * 100, 1)
        topic_tag = f'<span style="font-size:11px;color:#888;margin-left:8px">{row["topic"]}</span>'
        items += f"""
        <div style="display:flex;align-items:flex-start;gap:14px;padding:10px 0;border-bottom:1px solid #f0f0f0">
            <div style="font-size:22px;font-weight:800;color:{color};opacity:0.25;min-width:28px;line-height:1">{rank}</div>
            <div style="flex:1">
                <div style="font-size:14px;color:#1a1a2e;line-height:1.4;font-weight:500">{row['headline_text']}{topic_tag}</div>
                <div style="font-size:11px;color:#888;margin-top:3px">Outrage density: <strong>{score}</strong> per 100 words</div>
            </div>
        </div>"""

    return f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;padding:8px 0">
        <h4 style="font-size:13px;text-transform:uppercase;letter-spacing:0.07em;color:#666;margin:0 0 8px 0">
            Top {n} Most Outrageous Headlines
        </h4>
        {items}
    </div>"""


# ── NMF sub-theme stacked bar ──────────────────────────────────────────────────
NMF_COLORS = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A"]
TOPICS_ORDER = ["AI", "Climate", "Economy", "Immigration", "Iran"]

def nmf_chart(outlet):
    sub = nmf_props[nmf_props["outlet"] == outlet].copy()
    tcols = [c for c in sub.columns if c.startswith("T")]

    fig = go.Figure()
    for i, t in enumerate(tcols):
        labels = [
            nmf_labels.get(topic, {}).get(t, t)
            for topic in sub["topic"]
        ]
        fig.add_trace(go.Bar(
            name=t,
            x=sub["topic"],
            y=sub[t],
            marker_color=NMF_COLORS[i],
            opacity=0.85,
            hovertext=labels,
            hovertemplate="%{hovertext}<br>Proportion: %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Sub-Theme Mix by Topic (NMF)", x=0.5, font=dict(size=13)),
        barmode="stack",
        xaxis=dict(categoryorder="array", categoryarray=TOPICS_ORDER),
        yaxis=dict(title="Proportion", range=[0, 1]),
        legend=dict(title="Sub-theme", orientation="v", x=1.02, y=1.0, font=dict(size=10)),
        height=380,
        margin=dict(t=60, b=40, l=50, r=140),
    )
    return fig

def generate_topic_charts(outlet):
    return sentiment_by_topic_chart(outlet), outrage_by_topic_chart(outlet)


# ── Comparison: NRC radar overlay ──────────────────────────────────────────────
def compare_radar(selected_outlets, topic):
    fig = go.Figure()
    for o in selected_outlets:
        color = OUTLET_COLORS[o]
        if topic == "All":
            body_row = _row(nrc_body, outlet=o)
            head_row = _row(nrc_headline, outlet=o)
        else:
            body_row = _row(nrc_body_t, outlet=o, topic=topic)
            head_row = _row(nrc_head_t, outlet=o, topic=topic)
        for label, row, dash in [("Body", body_row, "solid"), ("Headline", head_row, "dot")]:
            if row is None:
                continue
            vals = [float(row[e]) for e in EMOTIONS] + [float(row[EMOTIONS[0]])]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=EMOTIONS + [EMOTIONS[0]],
                mode="lines", name=f"{o} ({label})",
                line=dict(color=color, dash=dash, width=2),
            ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, RADAR_MAX], tickfont=dict(size=9))),
        title=dict(text="Emotion Fingerprint Comparison", x=0.5, font=dict(size=13)),
        legend=dict(orientation="v", x=1.05, y=1.0, font=dict(size=11)),
        height=480,
        margin=dict(t=60, b=40, l=60, r=160),
    )
    return fig


# ── Comparison: MFT grouped bar ────────────────────────────────────────────────
def compare_mft(selected_outlets, topic):
    # mft.csv is pre-×1000; mft_topic.csv stores raw ratios
    scale = 1 if topic == "All" else 1000

    fig = go.Figure()
    for o in selected_outlets:
        if topic == "All":
            row = _row(mft, outlet=o)
        else:
            row = _row(mft_topic, outlet=o, topic=topic)
        if row is None:
            continue
        fig.add_trace(go.Bar(
            name=o,
            x=[MFT_LABELS[d] for d in MFT_DIMS],
            y=[float(row[d]) * scale for d in MFT_DIMS],
            marker_color=OUTLET_COLORS[o],
            opacity=0.85,
        ))

    topic_suffix = "" if topic == "All" else f" — {topic}"
    fig.update_layout(
        title=dict(text=f"Moral Foundations by Outlet (per 1,000 words){topic_suffix}", x=0.5, font=dict(size=13)),
        barmode="group",
        yaxis=dict(title="Score"),
        legend=dict(orientation="h", y=1.12),
        height=360,
        margin=dict(t=70, b=40, l=50, r=20),
    )
    return fig


# ── Comparison: Sentiment & outrage summary ────────────────────────────────────
def compare_sentiment(selected_outlets, topic):
    # Sentiment — topic-aware
    if topic == "All":
        body_vals     = [float(_row(sent_body,     outlet=o)["compound"]) for o in selected_outlets]
        headline_vals = [float(_row(sent_headline, outlet=o)["compound"]) for o in selected_outlets]
    else:
        st_rows = [_row(sent_topic, outlet=o, topic=topic) for o in selected_outlets]
        body_vals     = [float(r["body_compound"])     if r is not None else 0.0 for r in st_rows]
        headline_vals = [float(r["headline_compound"]) if r is not None else 0.0 for r in st_rows]

    # Outrage gap — topic-aware
    if topic == "All":
        outrage_vals = [float(_row(outrage, outlet=o)["outrage_gap"]) for o in selected_outlets]
    else:
        body_t = nrc_body_t[nrc_body_t["topic"] == topic].set_index("outlet")
        head_t = nrc_head_t[nrc_head_t["topic"] == topic].set_index("outlet")
        og_series = (head_t[OUTRAGE_EMOTIONS].sum(axis=1)
                     - body_t[OUTRAGE_EMOTIONS].sum(axis=1))
        outrage_vals = [float(og_series.loc[o]) if o in og_series.index else 0.0 for o in selected_outlets]

    topic_suffix = "" if topic == "All" else f" — {topic}"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Body sentiment", x=selected_outlets,
        y=body_vals,
        marker_color=[OUTLET_COLORS[o] for o in selected_outlets],
        opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        name="Headline sentiment", x=selected_outlets,
        y=headline_vals,
        marker_color=[OUTLET_COLORS[o] for o in selected_outlets],
        opacity=0.45,
    ))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=dict(text=f"Sentiment Comparison (VADER compound){topic_suffix}", x=0.5, font=dict(size=13)),
        barmode="group",
        yaxis=dict(title="Compound score"),
        legend=dict(orientation="h", y=1.12),
        height=320,
        margin=dict(t=70, b=40, l=50, r=20),
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="Outrage gap", x=selected_outlets,
        y=outrage_vals,
        marker_color=[OUTLET_COLORS[o] for o in selected_outlets],
        opacity=0.85,
        text=[f"{v:.2f}" for v in outrage_vals],
        textposition="outside",
    ))
    fig2.update_layout(
        title=dict(text=f"Headline Outrage Gap by Outlet (per 100 words){topic_suffix}", x=0.5, font=dict(size=13)),
        yaxis=dict(title="Gap (headline − body)"),
        height=320,
        margin=dict(t=70, b=40, l=50, r=20),
    )
    return fig, fig2


# ── Comparison: entity mention chart ──────────────────────────────────────────
def compare_entities(selected_outlets, etype, topic, n=12):
    sub = ent_df[ent_df["entity_type"] == etype]
    if topic != "All":
        sub = sub[sub["topic"] == topic]
    sub = sub[sub["outlet"].isin(selected_outlets)]

    # top N entities by total mentions across selected outlets
    top_entities = (
        sub.groupby("entity")["entity"].count()
        .nlargest(n).index.tolist()
    )

    fig = go.Figure()
    for o in selected_outlets:
        counts = sub[sub["outlet"] == o]["entity"].value_counts()
        fig.add_trace(go.Bar(
            name=o,
            x=top_entities,
            y=[int(counts.get(e, 0)) for e in top_entities],
            marker_color=OUTLET_COLORS[o],
            opacity=0.85,
        ))

    fig.update_layout(
        title=dict(text=f"Top {n} {etype} Mentions by Outlet", x=0.5, font=dict(size=13)),
        barmode="group",
        xaxis=dict(tickangle=-30, tickfont=dict(size=11)),
        yaxis=dict(title="Mentions"),
        legend=dict(orientation="h", y=1.12),
        height=400,
        margin=dict(t=70, b=100, l=50, r=20),
    )
    return fig


def generate_comparison(selected_outlets, topic):
    if not selected_outlets:
        empty = go.Figure()
        return empty, empty, empty, empty
    sent_fig, outrage_fig = compare_sentiment(selected_outlets, topic)
    return compare_radar(selected_outlets, topic), compare_mft(selected_outlets, topic), sent_fig, outrage_fig


def generate_entity_comparison(selected_outlets, etype, topic):
    if not selected_outlets:
        return go.Figure()
    return compare_entities(selected_outlets, etype, topic)


# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="News Outlet Profiles", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.HTML("""
    <div style="font-family:system-ui,-apple-system,sans-serif;padding:24px 0 8px 0;border-bottom:2px solid #eee;margin-bottom:8px">
        <h1 style="font-size:36px;font-weight:800;color:#1a1a2e;margin:0;letter-spacing:-1px">
            News Outlet Profile Explorer
        </h1>
        <p style="font-size:15px;color:#666;margin:6px 0 0 0;font-weight:400">
            How do CNN, Fox News, BBC, AP, and Newsmax frame the news differently?
            Select an outlet tab and filter by topic to explore tone, emotion, moral language, and entity focus.
        </p>
    </div>
    """)

    with gr.Tabs():
        for outlet in OUTLETS:
            with gr.Tab(outlet):
                topic_dd = gr.Dropdown(
                    choices=TOPICS, value="All", label="Filter by topic", scale=1
                )

                summary_md  = gr.HTML()

                with gr.Row():
                    radar_plot = gr.Plot(label="Emotional Fingerprint")
                    mft_plot   = gr.Plot(label="Moral Framing")

                terms_md       = gr.Markdown()
                headlines_html = gr.HTML()

                etype_dd = gr.Dropdown(
                    choices=ENTITY_TYPES, value="PERSON", label="Entity type"
                )
                entities_plot = gr.Plot()

                gr.Markdown("---\n#### Across All Topics")
                with gr.Row():
                    sent_topic_plot    = gr.Plot(label="Sentiment by Topic")
                    outrage_topic_plot = gr.Plot(label="Outrage Density by Topic")

                nmf_plot = gr.Plot(label="Sub-Theme Mix by Topic")

                def make_profile_fn(o):
                    def fn(topic):
                        return generate_profile(o, topic)
                    return fn

                def make_entity_fn(o):
                    def fn(etype, topic):
                        return generate_entity_tab(o, etype, topic)
                    return fn

                def make_topic_fn(o):
                    def fn():
                        return generate_topic_charts(o)
                    return fn

                profile_outputs  = [summary_md, radar_plot, mft_plot, terms_md]
                entity_outputs   = [entities_plot]
                topic_outputs    = [sent_topic_plot, outrage_topic_plot]
                headline_outputs = [headlines_html]

                def make_headline_fn(o):
                    def fn(topic):
                        return top_headlines_html(o, topic)
                    return fn

                topic_dd.change(fn=make_profile_fn(outlet), inputs=topic_dd, outputs=profile_outputs)
                topic_dd.change(fn=make_entity_fn(outlet), inputs=[etype_dd, topic_dd], outputs=entity_outputs)
                topic_dd.change(fn=make_headline_fn(outlet), inputs=topic_dd, outputs=headline_outputs)
                etype_dd.change(fn=make_entity_fn(outlet), inputs=[etype_dd, topic_dd], outputs=entity_outputs)
                demo.load(fn=lambda o=outlet: generate_profile(o, "All"), outputs=profile_outputs)
                demo.load(fn=lambda o=outlet: generate_entity_tab(o, "PERSON", "All"), outputs=entity_outputs)
                demo.load(fn=lambda o=outlet: top_headlines_html(o, "All"), outputs=headline_outputs)
                demo.load(fn=make_topic_fn(outlet), outputs=topic_outputs)
                demo.load(fn=lambda o=outlet: nmf_chart(o), outputs=nmf_plot)

        # ── Compare tab ───────────────────────────────────────────────────────
        with gr.Tab("Compare Outlets"):
            gr.Markdown("### Compare outlets side by side — select any combination below.")
            with gr.Row():
                outlet_checks = gr.CheckboxGroup(
                    choices=OUTLETS, value=OUTLETS, label="Outlets to include"
                )
                compare_topic = gr.Dropdown(
                    choices=TOPICS, value="All", label="Filter by topic"
                )

            with gr.Row():
                cmp_radar   = gr.Plot(label="Emotion Fingerprint Overlay")
                cmp_mft     = gr.Plot(label="Moral Foundations")

            with gr.Row():
                cmp_sent    = gr.Plot(label="Sentiment")
                cmp_outrage = gr.Plot(label="Outrage Gap")

            gr.Markdown("---\n#### Entity Mentions")
            etype_dd = gr.Dropdown(
                choices=ENTITY_TYPES, value="PERSON", label="Entity type"
            )
            cmp_entities = gr.Plot()

            cmp_outputs     = [cmp_radar, cmp_mft, cmp_sent, cmp_outrage]
            entity_inputs   = [outlet_checks, etype_dd, compare_topic]

            outlet_checks.change(fn=generate_comparison, inputs=[outlet_checks, compare_topic], outputs=cmp_outputs)
            outlet_checks.change(fn=generate_entity_comparison, inputs=entity_inputs, outputs=cmp_entities)
            compare_topic.change(fn=generate_comparison, inputs=[outlet_checks, compare_topic], outputs=cmp_outputs)
            compare_topic.change(fn=generate_entity_comparison, inputs=entity_inputs, outputs=cmp_entities)
            etype_dd.change(fn=generate_entity_comparison, inputs=entity_inputs, outputs=cmp_entities)
            demo.load(fn=lambda: generate_comparison(OUTLETS, "All"), outputs=cmp_outputs)
            demo.load(fn=lambda: generate_entity_comparison(OUTLETS, "PERSON", "All"), outputs=cmp_entities)

if __name__ == "__main__":
    demo.launch()