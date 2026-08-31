# SpectroBench: Spezifikation

Ein kontrollierter Benchmark zur Frage, ob und wo Fourier-basierte Repräsentationen lokaler Faltung (CNN) und klassischen Methoden überlegen sind, mit dem eigentlichen Ziel, nicht einen Sieger auszurufen, sondern die Regime-Grenze zu kartieren.

Version 0.1, Arbeitsname SpectroBench (Platzhalter, frei umbenennbar).

## 1. Kernthese und Motivation

Die Ausgangsintuition lautet, dass Fourier-Transformation und Fourier-Reihe im Machine Learning unterschätzt werden und dass klassisches pattern matching im Kern eine spektrale Operation ist. Diese These ist attraktiv, aber in ihrer naiven Form ("Fourier ist genauer als CNN auf PDEs") bereits erschöpfend publiziert und damit kein Beitrag. Der Beitrag dieses Benchmarks liegt an einer anderen Stelle, die die vorhandenen Suiten nicht sauber besetzen: eine spektral aufgelöste, budgetkontrollierte Vermessung, die pro Frequenzband misst, wo der Fourier-Vorteil real ist, wo er durch das Gibbs-Phänomen kippt, und die zeigt, dass ein einzelner Fehlerskalar genau diese Information verbirgt.

Drei Fourier-Konstrukte werden bewusst getrennt gehalten, weil sie in der Debatte ständig verwechselt werden und an verschiedenen Stellen des Netzes wirken:

1. Spektrale Faltungsschicht (Fourier Neural Operator, FNO): globale Faltung als Multiplikation im Frequenzraum, mit Modentruncation als sensiblem Knopf.
2. Fourier-Reihe beziehungsweise Fourier-Features als Koordinaten-Encoding (Tancik 2020, Wang und Perdikaris 2021): Basis gegen den spectral bias eines MLP.
3. Klassische Korrelation als pattern matching (Korrelationstheorem): eine Multiplikation zweier Spektren.

Der Benchmark testet alle drei gegen ihre lokalen beziehungsweise klassischen Gegenstücke.

## 2. Forschungsfrage und Abgrenzung

Leitfrage: Unter welchen messbaren Bedingungen (Glattheit des Ziels, geforderte Auflösungsgeneralisierung, Datenmenge, Frequenzgehalt des Musters) gewinnt eine Fourier-Repräsentation gegen lokale Faltung, und wo invertiert sich der Vorteil?

Abgrenzung zum Stand der Technik:

- PDEBench (Takamoto et al. 2022) und PINNacle liefern PDE-Aufgaben und Standardmetriken, berichten aber im Kern skalare Fehler und keine bandaufgelöste, energiegewichtete Zerlegung und keine kontrollierte Regime-Grenzkurve über einen Diskontinuitätsknopf.
- GFNet, FNet und verwandte Fourier-Mixing-Arbeiten zeigen globale spektrale Mischung in Vision und Sprache, aber nicht als kontrollierten Vergleich gegen lokale Faltung bei angeglichenem Budget mit Frequenz-Sweep.

Der Neuheitswinkel ist also die Kombination aus bandaufgelöster Metrik, kontrolliertem Diskontinuitätsknopf und der expliziten Brücke von pattern matching zum Korrelationstheorem, alles unter angeglichenem Parameter- und Rechenbudget.

## 3. Vorregistrierte Hypothesen

Jede Hypothese nennt die entscheidende Metrik und das Falsifikationskriterium vor dem Lauf. Diese Vorregistrierung ist bewusst im Geist der Absicherungslogik gehalten: das Bestehens- beziehungsweise Versagenskriterium steht fest, bevor Daten gesehen werden.

**H1a Spektraltreue.** Bei angeglichenem Budget rekonstruiert die Fourier-Repräsentation das hochfrequente Band mit kleinerem absolutem Bandfehler als das CNN, gemessen an `band_abs` in den oberen Bändern, dort wo `band_energy_frac` nicht vernachlässigbar ist. Falsifikation: liegen die oberen `band_abs` innerhalb der Seed-Streuung gleich oder ist der Vorteil kleiner als die gepaarte Standardabweichung, gilt H1a für diesen Datensatz als widerlegt.

**H1b Auflösungsgeneralisierung.** Trainiert auf Gitter R0, zero-shot getestet auf 2 R0 und 4 R0, wächst der relative Fehler der Fourier-Repräsentation deutlich langsamer als der des CNN. Entscheidungsmetrik: Steigung der Kurve rel_l2 über Testauflösung. Falsifikation: gleiche oder steilere Degradation der Fourier-Variante.

**H1c Dateneffizienz.** Auf glatten, quasi-periodischen Feldern erreicht die Fourier-Repräsentation ein Zielfehlerniveau mit weniger Trainingsbeispielen. Entscheidungsmetrik: Fehler über Trainingsgröße (log-log), verglichen über die Approximationsrate (Exponent). Falsifikation: kein signifikanter Ratenunterschied.

**H1d Regime-Grenze.** Es existiert ein messbarer Diskontinuitätsschwellwert (Burgers-Viskosität nu beziehungsweise Kantenschärfe), jenseits dessen sich der Fourier-Vorteil invertiert (Gibbs). Entscheidungsmetrik: Vorteilskennzahl (Fehlerverhältnis CNN zu Fourier) als Funktion von nu, Nulldurchgang nu-Stern. Ergebnis ist die Lokalisierung von nu-Stern selbst, nicht ein Sieg. Diese Hypothese kann nicht falsifiziert werden im üblichen Sinn, sie ist eine Messung, ihr Beitrag ist die Grenzkarte.

**H2 Pattern-Matching-These.** In einer Lokalisierungsaufgabe erreicht die klassische FFT-Kreuzkorrelation bei hohem SNR nahezu perfekte Lokalisierung, und ein lernendes spektrales Mischmodell übertrifft ein lokales CNN gleichen Budgets bei global-periodischen Mustern, insbesondere wenn die diskriminierende Information eine langreichweitige Phasenbeziehung ist. Entscheidungsmetriken: Lokalisierungsfehler über SNR, Parameter-zu-Genauigkeit bei Gitterfrequenz-Klassifikation.

## 4. Metriken

Alle Feldmetriken sind in `spectrobench/metrics.py` implementiert und batchfähig.

**Relativer L2-Fehler (rel_l2).** Der Standard und die Referenzzahl, absichtlich nur zusammen mit der Bandzerlegung berichtet, nie allein. Er ist von den energiereichen tiefen Frequenzen dominiert und verbirgt den hochfrequenten Fehler.

**Bandaufgelöster Fehler.** Radiale Frequenzbänder (1D und 2D) über eine fftshift-Zerlegung. Drei Größen pro Band:

- `relative`: Bandfehler geteilt durch Bandenergie des Ziels. Empfindlich, aber schlecht konditioniert in nahezu leeren Bändern, wo der Nenner gegen null geht und der relative Fehler explodiert. Nur im Verbund mit `energy_frac` lesen.
- `absolute`: Bandfehler normiert auf die Gesamtnorm des Ziels. Bänder sind vergleichbar, leere Bänder bleiben nahe null. Dies ist die robuste Kopfzahl.
- `energy_frac`: Anteil der Zielspektralenergie im Band, also wie sehr ein Band überhaupt zählt und wie vertrauenswürdig sein relativer Fehler ist.

**Radiales Leistungsspektrum (radial_psd) und log-spektrale Distanz.** E(k) radial gemittelt, plus der skaleninvariante Abstand der log-Spektren als Maß, wie gut die Form der Energiekaskade getroffen wird, unabhängig von der Gesamtmagnitude. Für Turbulenz die physikalisch entscheidende Größe.

**Rechenbudget.** Parameterzahl, Trainings-FLOPs beziehungsweise Wandzeit auf fester Hardware, Inferenzlatenz. Vergleiche werden als Pareto-Front Fehler über Rechenaufwand berichtet, nicht als Einzelpunkte.

## 5. Track A: PDE-Surrogat

Aufgabe: Operatorlernen, Abbildung Anfangsfeld u0 auf Lösung u(T). Gitterbasiert für CNN und FNO, koordinatenbasiert für die Fourier-Feature-Variante.

Datensatzfamilie mit einem Glattheitsknopf, damit die Grenze kartierbar ist:

| Datensatz | Regime | Rolle |
| --- | --- | --- |
| Diffusion (Wärmeleitung) | glatt, niederfrequent | Sanity-Anker, Fourier darf nur mild gewinnen |
| Burgers, nu-Sweep | glatt bis steile Front bis Schock | Hauptknopf für H1d |
| Advektion, unstetige Anfangsprofile (Rechteck) | reiner Gibbs-Stress | oberes Ende der Grenzkarte |
| Navier-Stokes 2D (Vortizität), optional | mehrskalig, Turbulenz | E(k)-Treue, log-spektrale Distanz |

Der Burgers-Löser ist ein Integrating-Factor-RK4 (Cox-Matthews) im Fourier-Raum mit 2/3-Dealiasing. Die Diffusion wird exakt über exp(-nu k^2 dt) integriert, was die Steifheit entfernt, die ein naives explizites Schema sprengt. Wichtig als Befund und als Threat: sehr kleines nu erfordert höhere Gitterauflösung, um schockaufgelöst zu bleiben. Diese Auflösungsforderung ist selbst Teil der Regime-Grenze und wird mitberichtet, nicht wegkalibriert.

Modelle (Rollen):

- CNN-Baseline: kleines U-Net beziehungsweise dilatierte 1D-Faltung, ausschließlich kleine lokale Kernel. Der ehrliche lokale Gegner.
- FNO1d, FNO2d: spektrale Faltung mit Modentruncation.
- Fourier-Feature-MLP: Koordinatennetz mit fester zufälliger Fourier-Basis, für die maschenfreie Sicht.
- Plain-MLP: spectral-bias-Baseline, die die Fourier-Features reparieren sollen.
- Klassischer Löser: als Referenzwahrheit und als Vergleichspunkt gelernt gegen klassisch (Genauigkeit mit Garantie gegen amortisierte Inferenzgeschwindigkeit, siehe Threats).

Splits: In-Distribution, Auflösungs-OOD (feineres Gitter, H1b), Parameter-OOD (nu-Werte außerhalb des Trainingsbands, Extrapolation).

## 6. Track B: Pattern Matching

Ziel ist, die These pattern matching ist indirekt Fourier quantitativ und falsifizierbar zu machen. Drei geerdete Teilaufgaben, alle synthetisch und kontrollierbar, ohne Downloads.

**B1 Gitterfrequenz-Regression.** 2D-Sinusgitter mit kontrollierter Ortsfrequenz, Orientierung und Phase, plus Rauschen. Ziel ist die Frequenz. Das ist eine rein spektrale Frage. Ein lokales CNN muss über ein großes rezeptives Feld integrieren, um ein globales periodisches Muster zu sehen, ein Fourier-Mischmodell liest es aus einer Transformation ab. Der Knopf ist die Frequenz Richtung Nyquist. Vorhersage: CNN-Genauigkeit fällt zu hohen Frequenzen, spektral bleibt flach.

**B2 Template-Lokalisierung.** Ein bekanntes Template wird an zufälliger Position in Rauschen platziert, Ziel ist der Ort. Das ist wörtlich Kreuzkorrelation, die nach dem Korrelationstheorem eine Multiplikation im Fourier-Raum ist. Vergleich: klassische FFT-Korrelation gegen gelernt lokal gegen gelernt spektral, Genauigkeit über SNR und über Rechenaufwand. Dies ist der direkteste Beleg der These.

**B3 Globale Phasenbeziehung.** Eine Aufgabe, deren diskriminierendes Merkmal eine langreichweitige periodische Beziehung ist (zwei Gitter müssen über das ganze Bild phasengleich sein). Lokales CNN braucht Tiefe, Fourier bekommt es in einer Schicht. Gemessen als Parameter-zu-Genauigkeit.

Nebenbefund, der die Rausch-Randnotiz aus der Diskussion anschließt: Klassifikation unter additivem gaußschem Rauschen über den SNR, um die spektrale Trennbarkeit (Signal konzentriert, weißes Rauschen flach, SNR-Gradient über die Frequenz) messbar zu machen. Dies wird als Zusatzachse geführt, nicht als Kernhypothese, weil adversariales Rauschen ausdrücklich später vertieft wird.

## 7. Kontrollprotokoll

Ein Benchmark, der einer Architektur mehr Kapazität lässt, misst Kapazität, nicht Architektur. Daher:

- Angeglichenes Parameterbudget innerhalb von fünf Prozent (Binärsuche über die Breite, `budget.match_width`).
- Angeglichenes Trainings-Schritt- beziehungsweise FLOP-Budget, identischer Optimierer, identischer Zeitplan, identische Datensplits und Normalisierung. Wird der Zeitplan später pro Modell abgestimmt, wird das transparent berichtet.
- Mindestens fünf Seeds, Bericht als Mittel plus Standardabweichung, gepaarte Tests über Seeds hinweg.
- Explizit definierte OOD-Splits (Auflösung, Parameter).

## 8. Ablationen

Der Fourier-Vorteil hat einen sensiblen Knopf, und ein ehrlicher Benchmark charakterisiert ihn, statt ihn zu cherry-picken.

- FNO-Modenzahl: Fehler und Stabilität über die Zahl behaltener Moden. Zu viele Moden übertragen hochfrequentes Rauschen und destabilisieren, in einem PDE-Residuum multipliziert die Ableitung die Moden mit dem Wellenzahlbetrag hoch der Ableitungsordnung.
- Fourier-Feature-Bandbreite sigma: der analoge Knopf im Koordinatennetz, Sweep über sigma mit der NTK-Begründung im Hintergrund.

## 9. Auswertung und Reporting

Kopf-Artefakte: die Pareto-Front Fehler über Rechenaufwand, die Bandfehler-Kurve `band_abs` mit hinterlegtem `energy_frac`, die Auflösungsgeneralisierungskurve, die Dateneffizienzkurve und die Regime-Grenzkurve mit nu-Stern. Jede Hypothese wird gegen ihr vorregistriertes Kriterium abgeschlossen mit klarer Aussage bestanden, widerlegt oder gemessen.

## 10. Threats to Validity

- Bandrelativer Fehler ist in nahezu leeren Bändern schlecht konditioniert. Deshalb ist die robuste Kopfzahl `band_abs`, gelesen zusammen mit `energy_frac`. Der Smoke-Test zeigt diesen Effekt live, die oberen Bänder tragen dort unter ein Prozent der Energie.
- Spektralgehalt der Daten muss kalibriert sein. Ein glattes Ziel (steiler Spektralabfall) hat schlicht keine hochfrequente Energie, dann ist die Spektraltreue-Hypothese nicht testbar. Datensätze müssen so gewählt sein, dass die oberen Bänder messbar besetzt sind (flacherer Anfangsspektrum, kleineres nu, echte Turbulenz).
- Schockregime und Löserauflösung sind gekoppelt. Sehr kleines nu braucht feineres Gitter, sonst ist die Referenzwahrheit selbst unteraufgelöst. Die Auflösungsforderung wird mitberichtet.
- Gelernt gegen klassisch ist keine Genauigkeitsfrage allein. Der klassische Löser hat Konvergenzordnung und Fehlerschranken, das gelernte Modell gewinnt nur in amortisierter Inferenzgeschwindigkeit. Beide Achsen werden getrennt berichtet.
- Optimierung kann Architektur überdecken. Ein großer Teil der PINN- und Surrogat-Literatur zeigt, dass Verlustgewichtung und Zeitplan ähnlich wirksam sind wie die Basis. Daher das feste, geteilte Trainingsprotokoll.

## 11. Repo-Struktur und Ausführung

```
spectrobench/
  SPEC.md                 dieses Dokument
  README.md               Kurzanleitung
  requirements.txt        torch, numpy
  run_smoke.py            Mini-Ende-zu-Ende-Lauf (CPU, ein bis zwei Minuten)
  run_experiment.py       konfigurierbarer Grid-Treiber
  spectrobench/
    metrics.py            rel_l2, bandaufgelöster Fehler, radiales PSD, log-spektrale Distanz
    budget.py             Parameterzählung und Breiten-Matching
    train.py              geteilter Trainings- und Evaluationsloop
    data/pde.py           Burgers (IFRK4), Diffusion, gaußsche Zufallsfelder
    data/patterns.py      Gitter, Template-Lokalisierung, FFT-Korrelation
    models/fno.py         SpectralConv1d/2d, FNO1d, FNO2d
    models/conv.py        CNN1d, kleines U-Net (lokale Kernel)
    models/fourier_mlp.py Fourier-Feature-MLP, Plain-MLP
```

Ausführen: `pip install -r requirements.txt` und dann `python run_smoke.py` für den Schnelltest oder `python run_experiment.py` für den konfigurierbaren Lauf.

## 12. Stand des Smoke-Tests

Bereits lauffähig und verifiziert auf CPU:

- Track A, Burgers bei nu gleich sechs mal zehn hoch minus drei, FNO1d gegen CNN1d bei rund 42 Tausend Parametern beidseitig. In-Distribution liegt das CNN sogar leicht vorn (rel_l2 0.057 gegen 0.063), ein ehrliches Ergebnis, das zeigt, dass Fourier bei steilen Fronten nicht automatisch gewinnt. Der entscheidende Diskriminator ist die Auflösungsgeneralisierung: FNO bleibt bei doppelter Auflösung praktisch stabil (0.063 auf 0.065), das CNN bricht ein (0.057 auf 0.208). Das ist H1b in Miniatur.
- Track B, Template-Lokalisierung per FFT-Korrelation: nahezu perfekt bei SNR zwei (Fehler 0.015 gegen 0.232 der Zentrums-Baseline), graceful degradation zu kleinem SNR. Das ist H2 in Miniatur und der direkte Beleg des Korrelationstheorems.

Der Smoke-Lauf ist bewusst klein (kleine Budgets, wenige Seeds) und ersetzt nicht den kontrollierten Lauf nach diesem Protokoll.

## 13. Referenzen

- Li et al., Fourier Neural Operator for Parametric PDEs, 2020. Li et al., Physics-Informed Neural Operator (PINO), 2021.
- Tancik et al., Fourier Features Let Networks Learn High Frequency Functions, 2020.
- Wang, Wang, Perdikaris, On the eigenvector bias of Fourier feature networks, 2021.
- Rahaman et al., On the Spectral Bias of Neural Networks, 2019.
- Takamoto et al., PDEBench, 2022. PINNacle, 2023.
- Rao et al., Global Filter Networks (GFNet), 2021. Lee-Thorp et al., FNet, 2021.
- Oppenheim und Lim, The importance of phase in signals, 1981.
- Cox und Matthews, Exponential Time Differencing for Stiff Systems, 2002 (Integrating-Factor-Schema).
