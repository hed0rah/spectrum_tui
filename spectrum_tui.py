#!/usr/bin/env python3
"""RTL-SDR Spectrum Scanner TUI"""

import asyncio
import subprocess
import struct
import tempfile
from pathlib import Path

import numpy as np
from textual.app import App
from textual.containers import Vertical, Horizontal
from textual.widgets import Header, Footer, Static, Label, Input, Button, DataTable
from textual.reactive import reactive
from textual.timer import Timer
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
        result = subprocess.run(
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
    """Renders a single-line spectrum bar chart."""

    spectrum: reactive[list[tuple[float, float]]] = reactive(list, always_update=True)

    def render_bar(self, values: list[float], width: int) -> str:
        if not values:
            return "No data"
        arr = np.array(values)
        # Bin into `width` columns
        n_bins = min(width, len(arr))
        if n_bins == 0:
            return "No data"
        binned = np.array_split(arr, n_bins)
        peaks = [np.max(b) for b in binned]

        lo, hi = min(peaks), max(peaks)
        if hi - lo < 1:
            hi = lo + 1
        bar = ""
        for p in peaks:
            idx = int((p - lo) / (hi - lo) * (len(BAR_CHARS) - 1))
            idx = max(0, min(idx, len(BAR_CHARS) - 1))
            bar += BAR_CHARS[idx]
        return bar

    def render(self) -> str:
        if not self.spectrum:
            return "[ waiting for scan data ]"
        powers = [p for _, p in self.spectrum]
        width = self.size.width - 2 if self.size.width > 4 else 40
        bar = self.render_bar(powers, width)
        freqs = [f for f, _ in self.spectrum]
        lo_label = freq_label(min(freqs))
        hi_label = freq_label(max(freqs))
        peak_freq, peak_db = max(self.spectrum, key=lambda x: x[1])

        return (
            f"{bar}\n"
            f" {lo_label:<20s} {'peak: ' + freq_label(peak_freq) + f' ({peak_db:.1f} dB)':^30s} {hi_label:>20s}"
        )


class PeakTable(DataTable):
    """Table of strongest signals found."""
    pass


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
    #spectrum-box {
        height: 5;
        padding: 0 1;
        border: round $primary;
    }
    #peak-table {
        height: 1fr;
        border: round $secondary;
    }
    """
    TITLE = "RTL-SDR Spectrum Scanner"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "scan", "Scan"),
        ("r", "scan", "Rescan"),
    ]

    scanning = reactive(False)

    def compose(self):
        yield Header()
        with Horizontal(id="controls"):
            yield Label(" From:", id="lbl-from")
            yield Input(value="88", placeholder="MHz", id="start-freq")
            yield Label(" To:", id="lbl-to")
            yield Input(value="108", placeholder="MHz", id="stop-freq")
            yield Label(" Top:", id="lbl-top")
            yield Input(value="20", placeholder="N", id="top-n")
            yield Button("Scan", variant="primary", id="btn-scan")
        yield Static("Press [bold]s[/bold] to scan  |  q to quit", id="status")
        yield SpectrumBar(id="spectrum-bar")
        yield PeakTable(id="peak-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#peak-table", PeakTable)
        table.add_columns("Rank", "Frequency", "Power (dB)", "Bar")
        table.cursor_type = "row"

    def action_scan(self) -> None:
        if not self.scanning:
            self.run_scan()

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

        # Update spectrum bar
        bar = self.query_one("#spectrum-bar", SpectrumBar)
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
        noise_floor = np.percentile([p for _, p in results], 10)

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
            f"{n_above} active channels (>10 dB above noise)")
        self.scanning = False


if __name__ == "__main__":
    app = SDRScannerApp()
    app.run()
