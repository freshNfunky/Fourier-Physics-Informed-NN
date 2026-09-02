# Fourier-Physics-Informed-NN

Ein kontrollierter Benchmark zur Frage, wo Fourier-basierte Repräsentationen lokaler Faltung (CNN) und klassischen numerischen Methoden überlegen sind, und wo sich der Vorteil umkehrt. Das Ziel ist ausdrücklich nicht, einen Sieger auszurufen, sondern die Regime-Grenze zu vermessen, mit einer spektral aufgelösten Metrik, die sichtbar macht, was ein einzelner Fehlerskalar verbirgt.

Arbeitsname der Suite: SpectroBench.

![Fourier vs plain PINN: the Fourier network catches the fine detail the plain one misses](pinn/media/spectral_bias.gif)

*Sofort sichtbar: das Fourier-Netz faengt die feine Schwingung, das plain-Netz bleibt glatt und blind. Interaktive Demo unter `pinn/` und als HuggingFace Space in `space/`. Publikations- und Kanalplan in `docs/PUBLISHING.md`.*

## Motivation

Die Leitthese lautet, dass Fourier-Transformation und Fourier-Reihe im Machine Learning unterschätzt werden und dass klassisches pattern matching im Kern eine spektrale Operation ist. In ihrer naiven Form (Fourier ist genauer als CNN auf PDEs) ist diese These bereits erschöpfend publiziert und damit kein Beitrag. Der Beitrag dieses Repos liegt an drei Stellen, die die vorhandenen Suiten (PDEBench, PINNacle, GFNet) nicht sauber besetzen: der bandaufgelöste Fehler, die Auflösungsgeneralisierung und die Regime-Grenze über einen kontrollierten Diskontinuitätsknopf.

Drei Fourier-Konstrukte werden bewusst getrennt gehalten, weil sie in der Debatte ständig verwechselt werden:

1. Spektrale Faltungsschicht (Fourier Neural Operator, FNO): globale Faltung als Multiplikation im Frequenzraum, mit Modentruncation als sensiblem Knopf.
2. Fourier-Reihe beziehungsweise Fourier-Features als Koordinaten-Encoding (Tancik 2020, Wang und Perdikaris 2021): Basis gegen den spectral bias eines MLP.
3. Klassische Korrelation als pattern matching (Korrelationstheorem): eine Multiplikation zweier Spektren.

Der Benchmark testet alle drei gegen ihr lokales beziehungsweise klassisches Gegenstück, bei angeglichenem Parameter- und Rechenbudget.

## Was ist drin

- `SPEC.md` ist das inhaltliche Herz: Kernthese, vorregistrierte Hypothesen H1a bis H1d und H2 mit Entscheidungs- und Falsifikationskriterien, Metrikdefinitionen, beide Tracks, Kontrollprotokoll, Ablationen und Threats to Validity. Wer verstehen will, was gemessen wird und warum, liest dieses Dokument zuerst.
- Ein lauffähiges PyTorch-Gerüst, das die Achsen automatisch durchmisst. Auf CPU in ein bis zwei Minuten smoke-testbar.

## Schnellstart

```
pip install -r requirements.txt
python run_smoke.py
```

Der Schnelltest zeigt beide Tracks in Miniatur: Burgers-Operatorlernen (FNO gegen CNN, inklusive zero-shot Auflösungsgeneralisierung) und Template-Lokalisierung per FFT-Korrelation über den SNR.

Der kontrollierte Lauf:

```
python run_experiment.py --nus 3e-2 1e-2 6e-3 3e-3 --seeds 3 --steps 300
```

Viskositäts-Sweep (Regime-Grenze, H1d) und Auflösungsgeneralisierung (H1b) bei angeglichenem Parameterbudget, gemittelt über Seeds, mit Ausgabe nach `results.json`.

## Struktur

```
Fourier-Physics-Informed-NN/
  SPEC.md                 vollständige Spezifikation (Hypothesen, Metriken, Protokoll)
  README.md               dieses Dokument
  requirements.txt        torch, numpy
  run_smoke.py            Mini-Ende-zu-Ende-Lauf (CPU)
  run_experiment.py       konfigurierbarer Grid-Treiber
  spectrobench/
    metrics.py            rel_l2, bandaufgelöster Fehler, radiales PSD, log-spektrale Distanz
    budget.py             Parameterzählung und Breiten-Matching
    train.py              geteilter Trainings- und Evaluationsloop
    data/pde.py           Burgers (Integrating-Factor-RK4), Diffusion, gaußsche Zufallsfelder
    data/patterns.py      Gitter, Template-Lokalisierung, FFT-Korrelation
    models/fno.py         SpectralConv1d/2d, FNO1d, FNO2d
    models/conv.py        CNN1d, kleines U-Net (lokale Kernel)
    models/fourier_mlp.py Fourier-Feature-MLP, Plain-MLP
```

## Stand

Verifiziert auf CPU. Zwei ehrliche Frühbefunde aus dem Smoke-Test, bewusst nicht geglättet:

- In-Distribution auf steilen Fronten (Burgers, nu gleich sechs mal zehn hoch minus drei) gewinnt das CNN sogar leicht (rel_l2 0.057 gegen 0.063 des FNO). Fourier ist also nicht automatisch besser, genau der Gibbs-Punkt.
- Der entscheidende Diskriminator ist die Auflösungsgeneralisierung: das FNO bleibt bei doppelter Auflösung praktisch stabil (0.063 auf 0.065), das CNN bricht ein (0.057 auf 0.208). Das ist Hypothese H1b in Miniatur.
- Track B belegt das Korrelationstheorem direkt: FFT-Kreuzkorrelation lokalisiert bei SNR zwei nahezu perfekt (Fehler 0.015 gegen 0.232 der Zentrums-Baseline).

Eine methodische Feinheit ist bereits eingebaut: der bandrelative Fehler ist in fast leeren Bändern schlecht konditioniert, deshalb ist die robuste Kopfzahl der absolute Bandfehler zusammen mit dem Band-Energieanteil (siehe Threats to Validity in `SPEC.md`).

## Roadmap

- Vollständiger nu-Sweep mit Fehlerbalken, damit die Regime-Grenze nu-Stern belastbar herauskommt.
- Track B ausbauen: Gitterfrequenz-Regression und globale Phasenbeziehung als gelernte Vergleiche, nicht nur die klassische Korrelation.
- Navier-Stokes 2D für die Energiespektrum-Treue (radiales PSD, log-spektrale Distanz).
- Ablationen über FNO-Modenzahl und Fourier-Feature-Bandbreite sigma.

## Whitepaper

Ein IEEE-Whitepaper mit der vollstaendigen Dokumentation aller Benchmarks liegt unter `paper/`: `paper/whitepaper.tex` (Quelle, IEEEtran) und `paper/whitepaper.pdf` (kompiliert, vier Seiten). `paper/benchmarks.py` erzeugt die dokumentierten Zahlen, `paper/results.json` haelt sie als Provenance fest. Zahlen neu erzeugen mit `python paper/benchmarks.py`, Abbildungen mit `python paper/make_figures.py`, dann neu kompilieren mit `pdflatex paper/whitepaper.tex` (zweimal fuer Referenzen). Das Paper enthaelt ein TikZ-Architektur-Schema (Fig. 1) und die Ergebnis-Plots (Fig. 2).

- PINN-Domaene: physik-informierten Trainings-Track ergaenzen (PDE-Residualverlust via Autodiff) und benchmarken, ob der Fourier-Ansatz (Fourier-Feature-MLP bzw. PINO) ein reines PINN und die datengetriebenen Surrogate schlaegt, mit denselben band-, aufloesungs- und regimeaufgeloesten Metriken. Siehe `BACKLOG.md`.

## Referenzen

Siehe Abschnitt 13 in `SPEC.md`.
