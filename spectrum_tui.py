#!/usr/bin/env python3
"""RTL-SDR Spectrum Scanner TUI"""

import asyncio
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Static, Label, Input, Button, DataTable
from textual.reactive import reactive
from textual import work


# ── Helpers ──────────────────────────────────────────────────────────────

def freq_label(hz: float) -> str:
    """Human-readable frequency label."""
    if hz >= 1e9:
        return f"{hz / 1e9:.3f} GHz"
    if hz >= 1e6:
        return f"{hz / 1e6:.3f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:.1f} kHz"
    return f"{hz:.0f} Hz"


def capture_iq(center_freq: float, sample_rate: int = 2_048_000,
               num_samples: int = 262_144) -> np.ndarray | None:
    """Capture IQ samples using rtl_sdr CLI and return as complex array."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            ["rtl_sdr", "-f", str(int(center_freq)), "-s", str(sample_rate),
             "-n", str(num_samples), tmp_path],
            capture_output=True, timeout=10,
        )
        raw = Path(tmp_path).read_bytes()
        if len(raw) < 2:
            return None
        # RTL-SDR outputs interleaved unsigned 8-bit I/Q
        iq = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        iq = (iq - 127.5) / 127.5  # normalize to [-1, 1]
        return iq[0::2] + 1j * iq[1::2]
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def compute_spectrum(iq: np.ndarray, fft_size: int = 1024) -> np.ndarray:
    """Compute power spectrum in dB from IQ samples (averaged over segments)."""
    n_segments = len(iq) // fft_size
    if n_segments == 0:
        return np.zeros(fft_size)
    iq = iq[:n_segments * fft_size].reshape(n_segments, fft_size)
    window = np.hanning(fft_size)
    spectra = np.fft.fftshift(np.fft.fft(iq * window, axis=1), axes=1)
    power = np.mean(np.abs(spectra) ** 2, axis=0)
    power_db = 10 * np.log10(power + 1e-10)
    return power_db


def scan_band(start_hz: float, stop_hz: float, step_hz: float = 2_000_000,
              sample_rate: int = 2_048_000, fft_size: int = 1024,
              callback=None) -> list[tuple[float, float]]:
    """Sweep from start_hz to stop_hz, returning list of (freq_hz, power_db)."""
    results = []
    center = start_hz + sample_rate / 2
    while center - sample_rate / 2 < stop_hz:
        iq = capture_iq(center, sample_rate=sample_rate, num_samples=fft_size * 64)
        if iq is not None:
            spectrum = compute_spectrum(iq, fft_size)
            freqs = np.linspace(center - sample_rate / 2, center + sample_rate / 2,
                                fft_size, endpoint=False)
            for f, p in zip(freqs, spectrum):
                if start_hz <= f <= stop_hz:
                    results.append((float(f), float(p)))
        center += step_hz
        if callback:
            callback(center, stop_hz)
    results.sort(key=lambda x: x[0])
    return results


# ── Textual Widgets ──────────────────────────────────────────────────────

BAR_CHARS = " ▁▂▃▄▅▆▇█"


class SpectrumBar(Static):
    """Renders a single-line spectrum bar chart with noise floor marker."""

    spectrum: reactive[list[tuple[float, float]]] = reactive(list, always_update=True)
    noise_floor: reactive[float | None] = reactive(None)

    def _bar_index(self, value: float, lo: float, hi: float) -> int:
        idx = int((value - lo) / (hi - lo) * (len(BAR_CHARS) - 1))
        return max(0, min(idx, len(BAR_CHARS) - 1))

    def render(self) -> str:
        if not self.spectrum:
            return "[ waiting for scan data ]"

        powers = [p for _, p in self.spectrum]
        freqs = [f for f, _ in self.spectrum]
        width = self.size.width - 2 if self.size.width > 4 else 40

        # Bin into columns
        arr = np.array(powers)
        n_bins = min(width, len(arr))
        if n_bins == 0:
            return "No data"
        binned = np.array_split(arr, n_bins)
        peaks = [np.max(b) for b in binned]

        lo, hi = min(peaks), max(peaks)
        if hi - lo < 1:
            hi = lo + 1

        # Noise floor position in the bar range
        nf = self.noise_floor
        nf_idx = self._bar_index(nf, lo, hi) if nf is not None else None

        # Build the spectrum line with color: above noise = green, at/below = dim
        bar_line = ""
        for p in peaks:
            idx = self._bar_index(p, lo, hi)
            char = BAR_CHARS[idx]
            if nf is not None and p > nf + 6:
                bar_line += f"[bold green]{char}[/]"
            elif nf is not None and p > nf + 3:
                bar_line += f"[yellow]{char}[/]"
            else:
                bar_line += f"[dim]{char}[/]"

        # Noise floor ruler line — mark where it sits
        if nf is not None:
            nf_char = BAR_CHARS[nf_idx] if nf_idx is not None else "▁"
            ruler = ""
            nf_label = f"noise floor {nf:.0f} dB"
            # place the marker roughly in the middle with the label
            ruler = f"  [dim red]── {nf_char} {nf_label} ──[/]"
        else:
            ruler = ""

        lo_label = freq_label(min(freqs))
        hi_label = freq_label(max(freqs))
        peak_freq, peak_db = max(self.spectrum, key=lambda x: x[1])
        center_info = f"peak: {freq_label(peak_freq)} ({peak_db:.1f} dB)"

        return (
            f"{bar_line}\n"
            f" {lo_label:<20s} {center_info:^30s} {hi_label:>20s}\n"
            f"{ruler}"
        )


class PeakTable(DataTable):
    """Table of strongest signals found."""
    pass


class ZoomView(Static):
    """Single-frequency monitor — shows signal strength over time."""

    history: reactive[list[float]] = reactive(list, always_update=True)
    target_freq: reactive[float] = reactive(0.0)
    running: reactive[bool] = reactive(False)

    HISTORY_LEN = 60  # how many samples to keep

    def render(self) -> str:
        if not self.history:
            if self.running:
                return f"[yellow]Locking onto {freq_label(self.target_freq)}...[/]"
            return "[ select a frequency from the peak table and press [bold]z[/bold] to zoom ]"

        vals = self.history
        current = vals[-1]
        mn, mx = min(vals), max(vals)
        avg = sum(vals) / len(vals)

        # Spread for the chart
        chart_lo = mn - 2
        chart_hi = mx + 2
        if chart_hi - chart_lo < 4:
            chart_hi = chart_lo + 4

        width = self.size.width - 4 if self.size.width > 8 else 60

        # Time series bar — one column per sample, scaled to fit width
        if len(vals) > width:
            # downsample to fit
            step = len(vals) / width
            display_vals = [vals[int(i * step)] for i in range(width)]
        else:
            display_vals = vals

        line = ""
        for v in display_vals:
            idx = int((v - chart_lo) / (chart_hi - chart_lo) * (len(BAR_CHARS) - 1))
            idx = max(0, min(idx, len(BAR_CHARS) - 1))
            char = BAR_CHARS[idx]
            # color by strength relative to average
            if v > avg + 3:
                line += f"[bold green]{char}[/]"
            elif v > avg:
                line += f"[green]{char}[/]"
            elif v > avg - 3:
                line += f"[yellow]{char}[/]"
            else:
                line += f"[dim]{char}[/]"

        status = "[bold green]● LIVE[/]" if self.running else "[dim]● stopped[/]"
        delta = current - avg
        delta_color = "green" if delta > 0 else "red"

        return (
            f" {status}  [bold]{freq_label(self.target_freq)}[/]   "
            f"now: [bold]{current:.1f}[/] dB  "
            f"avg: {avg:.1f} dB  "
            f"Δ [{delta_color}]{delta:+.1f}[/]  "
            f"min: {mn:.1f}  max: {mx:.1f}\n"
            f" {line}\n"
            f" [dim]{'oldest':<{len(display_vals)//2}}{'newest':>{len(display_vals)//2}}[/]"
        )


# ── Main App ─────────────────────────────────────────────────────────────

class SDRScannerApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #controls {
        height: 3;
        layout: horizontal;
        padding: 0 1;
    }
    #controls Input {
        width: 20;
        margin: 0 1;
    }
    #controls Button {
        width: 16;
        margin: 0 1;
    }
    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #spectrum-bar {
        height: 5;
        padding: 0 1;
        border: round $primary;
    }
    #peak-table {
        height: 1fr;
        border: round $secondary;
    }
    #zoom-view {
        height: 6;
        padding: 0 1;
        border: round $success;
        display: none;
    }
    #zoom-view.active {
        display: block;
    }
    """
    TITLE = "RTL-SDR Spectrum Scanner"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "scan", "Scan"),
        ("r", "scan", "Rescan"),
        ("z", "zoom", "Zoom"),
        ("escape", "back", "Back"),
    ]

    scanning = reactive(False)
    zoom_active = reactive(False)
    _zoom_stop = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="controls"):
            yield Label(" From:", id="lbl-from")
            yield Input(value="88", placeholder="MHz", id="start-freq")
            yield Label(" To:", id="lbl-to")
            yield Input(value="108", placeholder="MHz", id="stop-freq")
            yield Label(" Top:", id="lbl-top")
            yield Input(value="20", placeholder="N", id="top-n")
            yield Button("Scan", variant="primary", id="btn-scan")
        yield Static("Press [bold]s[/bold] to scan  |  [bold]z[/bold] zoom into selected peak  |  q to quit", id="status")
        yield SpectrumBar(id="spectrum-bar")
        yield ZoomView(id="zoom-view")
        yield PeakTable(id="peak-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#peak-table", PeakTable)
        table.add_columns("Rank", "Frequency", "Power (dB)", "Bar")
        table.cursor_type = "row"

    def action_scan(self) -> None:
        if not self.scanning and not self.zoom_active:
            self.run_scan()

    def action_zoom(self) -> None:
        """Zoom into the currently selected frequency in the peak table."""
        if self.zoom_active or self.scanning:
            return
        table = self.query_one("#peak-table", PeakTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        row_data = table.get_row(row_key)
        # row_data: [rank, freq_label, power, bar_str]
        freq_str = row_data[1]
        # parse freq back to Hz
        freq_hz = self._parse_freq_label(freq_str)
        if freq_hz is None:
            return
        self._start_zoom(freq_hz)

    def action_back(self) -> None:
        """Exit zoom mode."""
        if self.zoom_active:
            self._stop_zoom()

    def _parse_freq_label(self, label: str) -> float | None:
        """Parse a freq_label() string back to Hz."""
        label = label.strip()
        try:
            if label.endswith("GHz"):
                return float(label.replace("GHz", "").strip()) * 1e9
            elif label.endswith("MHz"):
                return float(label.replace("MHz", "").strip()) * 1e6
            elif label.endswith("kHz"):
                return float(label.replace("kHz", "").strip()) * 1e3
            elif label.endswith("Hz"):
                return float(label.replace("Hz", "").strip())
        except ValueError:
            return None
        return None

    def _start_zoom(self, freq_hz: float) -> None:
        self.zoom_active = True
        self._zoom_stop = False
        zoom = self.query_one("#zoom-view", ZoomView)
        zoom.target_freq = freq_hz
        zoom.history = []
        zoom.running = True
        zoom.add_class("active")
        status = self.query_one("#status", Static)
        status.update(
            f"[bold green]ZOOM[/] {freq_label(freq_hz)}  |  "
            f"[bold]escape[/bold] to return to scanner"
        )
        self._run_zoom(freq_hz)

    def _stop_zoom(self) -> None:
        self._zoom_stop = True
        self.zoom_active = False
        zoom = self.query_one("#zoom-view", ZoomView)
        zoom.running = False
        zoom.remove_class("active")
        status = self.query_one("#status", Static)
        status.update(
            "Press [bold]s[/bold] to scan  |  "
            "[bold]z[/bold] zoom into selected peak  |  q to quit"
        )

    @work(thread=True)
    def _run_zoom(self, freq_hz: float) -> None:
        """Continuously sample a single frequency and update the zoom view."""
        zoom = self.query_one("#zoom-view", ZoomView)
        history = deque(maxlen=ZoomView.HISTORY_LEN)
        sample_rate = 2_048_000
        fft_size = 1024
        # We want the power at the center frequency
        while not self._zoom_stop:
            iq = capture_iq(freq_hz, sample_rate=sample_rate,
                            num_samples=fft_size * 32)
            if iq is not None:
                spectrum = compute_spectrum(iq, fft_size)
                # Take the peak power in the center ~200 kHz
                # Center bins: middle 10% of the FFT
                center_start = int(fft_size * 0.45)
                center_end = int(fft_size * 0.55)
                peak_power = float(np.max(spectrum[center_start:center_end]))
                history.append(peak_power)
                self.call_from_thread(setattr, zoom, "history", list(history))
        self.call_from_thread(setattr, zoom, "running", False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-scan":
            self.action_scan()

    @work(thread=True)
    def run_scan(self) -> None:
        self.scanning = True
        status = self.query_one("#status", Static)

        try:
            start_input = self.query_one("#start-freq", Input).value
            stop_input = self.query_one("#stop-freq", Input).value
            top_n_input = self.query_one("#top-n", Input).value

            start_hz = float(start_input) * 1e6
            stop_hz = float(stop_input) * 1e6
            top_n = int(top_n_input)
        except ValueError:
            self.call_from_thread(status.update, "[red]Invalid input — enter numbers[/red]")
            self.scanning = False
            return

        def progress(current, total):
            pct = min(100, int((current - start_hz) / (total - start_hz) * 100))
            self.call_from_thread(status.update,
                f"[yellow]Scanning... {freq_label(current)} ({pct}%)[/yellow]")

        self.call_from_thread(status.update, f"[yellow]Scanning {freq_label(start_hz)} → {freq_label(stop_hz)}...[/yellow]")

        results = scan_band(start_hz, stop_hz, callback=progress)

        if not results:
            self.call_from_thread(status.update, "[red]No data captured — is the dongle connected?[/red]")
            self.scanning = False
            return

        # Compute noise floor
        noise_floor = float(np.percentile([p for _, p in results], 10))

        # Update spectrum bar with noise floor
        bar = self.query_one("#spectrum-bar", SpectrumBar)
        self.call_from_thread(setattr, bar, "noise_floor", noise_floor)
        self.call_from_thread(setattr, bar, "spectrum", results)

        # Find peaks — simple approach: sort by power, pick top N with spacing
        sorted_by_power = sorted(results, key=lambda x: x[1], reverse=True)
        peaks = []
        min_spacing = 200_000  # 200 kHz minimum between peaks
        for freq, power in sorted_by_power:
            if all(abs(freq - pf) > min_spacing for pf, _ in peaks):
                peaks.append((freq, power))
            if len(peaks) >= top_n:
                break

        peaks.sort(key=lambda x: x[1], reverse=True)

        # Update table
        table = self.query_one("#peak-table", PeakTable)
        self.call_from_thread(table.clear)

        for i, (freq, power) in enumerate(peaks, 1):
            above_noise = power - noise_floor
            bar_len = max(0, int(above_noise / 2))
            bar_str = "█" * min(bar_len, 30)
            self.call_from_thread(
                table.add_row,
                str(i),
                freq_label(freq),
                f"{power:.1f}",
                bar_str,
            )

        n_above = sum(1 for _, p in results if p > noise_floor + 10)
        self.call_from_thread(status.update,
            f"[green]Done![/green] {len(results)} bins | "
            f"noise floor: {noise_floor:.1f} dB | "
            f"{n_above} active channels (>10 dB above noise) | "
            f"[bold]z[/bold] to zoom a peak")
        self.scanning = False


if __name__ == "__main__":
    app = SDRScannerApp()
    app.run()
