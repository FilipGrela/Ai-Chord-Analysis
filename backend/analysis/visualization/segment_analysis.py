"""Wizualizacja analizy segmentów, zwłaszcza rozkładu długości.

Moduł zawiera funkcje do generowania interaktywnych wykresów Plotly
opartych na danych segmentów (długość, rodzaj itp.).
"""

import numpy as np
import plotly.graph_objects as go

from backend.analysis.music_metrics import canonical_note_name, parse_chord


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
    fig.update_xaxes(range=[0, 40], autorange=False)
    
    # Wygeneruj HTML bez tagów <html>, <head>, <body> i zachowaj kolejność div+script.
    html_full = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="segment_duration_histogram",
    )

    # Zwracamy pusty skrypt i kompletne HTML (div + script) jako drugi element,
    # aby raport osadził go w poprawnej kolejności.
    return "", html_full


def generate_top_class_duration_barchart(
    class_durations: dict[str, float],
) -> tuple[str, str]:
    """Generuje wykres słupkowy top klas akordów wg czasu trwania.

    Args:
        class_durations: mapa klasy akordu -> łączny czas w sekundach.
        top_n: liczba klas do pokazania.

    Returns:
        Tuple (script_tag, div_tag) - skrypt plotly i div do osadzenia w HTML.
    """
    if not class_durations:
        return "", "<p>Brak danych do wykreślenia top klas.</p>"

    filtered = [(k, v) for k, v in class_durations.items() if not k.endswith(':other')]
    sorted_items = sorted(filtered, key=lambda item: item[1], reverse=True)
    labels = [label for label, _ in sorted_items]
    values = [value for _, value in sorted_items]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker=dict(color='rgba(60, 120, 180, 0.75)'),
        hovertemplate='<b>Class</b>: %{x}<br><b>Total</b>: %{y:.2f}s<extra></extra>',
    ))

    fig.update_layout(
        title=f"Top {len(labels)} Classes by Duration",
        xaxis_title="Chord Class",
        yaxis_title="Total Duration (s)",
        template="plotly_white",
        height=450,
        margin=dict(l=60, r=30, t=60, b=120),
    )
    fig.update_xaxes(tickangle=-45)

    html_full = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="top_class_duration_chart",
    )

    return "", html_full


def generate_transition_heatmap(
    class_labels: list[str],
    duration_matrix: np.ndarray,
    count_matrix: np.ndarray | None = None,
) -> tuple[str, str]:
    """Generuje heatmapę przejść między klasami akordów.

    Args:
        class_labels: uporządkowana lista etykiet klas na osiach.
        duration_matrix: macierz sumy czasów trwania przejść.
        count_matrix: opcjonalna macierz liczby przejść do hovera.

    Returns:
        Tuple (script_tag, div_tag) - skrypt plotly i div do osadzenia w HTML.
    """
    if duration_matrix.size == 0 or not class_labels:
        return "", "<p>Brak danych do wykreślenia heatmapy przejść.</p>"

    customdata = None
    hovertemplate = (
        "<b>From</b>: %{y}<br>"
        "<b>To</b>: %{x}<br>"
        "<b>Duration</b>: %{z:.2f}s<extra></extra>"
    )

    if count_matrix is not None and count_matrix.shape == duration_matrix.shape:
        total_transitions = float(np.sum(count_matrix))
        if total_transitions > 0:
            probabilities = count_matrix / total_transitions
        else:
            probabilities = np.zeros_like(count_matrix, dtype=float)
        customdata = np.dstack((count_matrix, probabilities))
        hovertemplate = (
            "<b>From</b>: %{y}<br>"
            "<b>To</b>: %{x}<br>"
            "<b>Duration</b>: %{z:.2f}s<br>"
            "<b>Count</b>: %{customdata[0]:.0f}<br>"
            "<b>Probability</b>: %{customdata[1]:.2%}<extra></extra>"
        )

    fig = go.Figure(
        data=go.Heatmap(
            z=duration_matrix,
            x=class_labels,
            y=class_labels,
            customdata=customdata,
            colorscale="Viridis",
            colorbar=dict(title="Duration (s)"),
            hovertemplate=hovertemplate,
        )
    )

    fig.update_layout(
        title="Transition Matrix Heatmap",
        xaxis_title="Chord N+1",
        yaxis_title="Chord N",
        template="plotly_white",
        height=900,
        margin=dict(l=110, r=30, t=70, b=180),
    )
    fig.update_xaxes(tickangle=-45, side="top")
    fig.update_yaxes(autorange="reversed")

    html_full = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="transition_matrix_heatmap",
    )

    return "", html_full


def transition_heatmap_label(chord: str) -> str:
    parsed = parse_chord(chord)
    canonical_root = canonical_note_name(parsed.root_pc)
    if parsed.quality == "N" or canonical_root is None:
        return "N"

    quality = "min" if parsed.is_minor is True else "maj"
    return f"{canonical_root}:{quality}"


def build_transition_heatmap_matrices(
    transition_counts: dict[str, int],
    transition_durations: dict[str, float],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    labels = [f"{note}:{quality}" for note in notes for quality in ("maj", "min")] + ["N"]
    index_by_label = {label: idx for idx, label in enumerate(labels)}
    count_matrix = np.zeros((len(labels), len(labels)), dtype=float)
    duration_matrix = np.zeros((len(labels), len(labels)), dtype=float)

    for transition, count in transition_counts.items():
        try:
            left, right = [part.strip() for part in transition.split("->", 1)]
        except ValueError:
            continue

        row_label = transition_heatmap_label(left)
        col_label = transition_heatmap_label(right)
        row_idx = index_by_label.get(row_label)
        col_idx = index_by_label.get(col_label)
        if row_idx is None or col_idx is None:
            continue
        count_matrix[row_idx, col_idx] += float(count)

    for transition, duration in transition_durations.items():
        try:
            left, right = [part.strip() for part in transition.split("->", 1)]
        except ValueError:
            continue

        row_label = transition_heatmap_label(left)
        col_label = transition_heatmap_label(right)
        row_idx = index_by_label.get(row_label)
        col_idx = index_by_label.get(col_label)
        if row_idx is None or col_idx is None:
            continue
        duration_matrix[row_idx, col_idx] += float(duration)

    return labels, count_matrix, duration_matrix
