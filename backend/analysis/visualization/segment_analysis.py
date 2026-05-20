"""Wizualizacja analizy segmentów, zwłaszcza rozkładu długości.

Moduł zawiera funkcje do generowania interaktywnych wykresów Plotly
opartych na danych segmentów (długość, rodzaj itp.).
"""

import plotly.graph_objects as go


def generate_segment_duration_graph(segment_durations: list[float], bin_size: float = 0.2) -> tuple[str, str]:
    """Generuje histogram rozkładu długości segmentów.
    
    Args:
        segment_durations: lista długości segmentów w sekundach.
        bin_size: szerokość przedziału histogramu (domyślnie 0.2 s).
    
    Returns:
        Tuple (script_tag, div_tag) - skrypt plotly i div do osadzenia w HTML.
    """
    if not segment_durations:
        return "", "<p>Brak danych do wykreślenia histogramu.</p>"
    
    # Oblicz statystyki
    min_dur = min(segment_durations)
    max_dur = max(segment_durations)
    mean_dur = sum(segment_durations) / len(segment_durations)
    
    # Utwórz histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=segment_durations,
        nbinsx=int((max_dur - min_dur) / bin_size) + 1,
        name="Segment Duration",
        marker=dict(color='rgba(100, 150, 200, 0.7)'),
        hovertemplate='<b>Duration Range</b>: %{x:.2f}s<br><b>Count</b>: %{y}<extra></extra>',
    ))
    
    # Dodaj pionową linię dla średniej
    fig.add_vline(
        x=mean_dur,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_dur:.2f}s",
        annotation_position="top right",
    )
    
    fig.update_layout(
        title="Segment Duration Distribution",
        xaxis_title="Duration (seconds)",
        yaxis_title="Count",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    
    # Wygeneruj HTML bez tagów <html>, <head>, <body> i zachowaj kolejność div+script.
    html_full = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="segment_duration_histogram",
    )

    # Zwracamy pusty skrypt i kompletne HTML (div + script) jako drugi element,
    # aby raport osadził go w poprawnej kolejności.
    return "", html_full
