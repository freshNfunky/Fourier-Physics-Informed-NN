# Publication and distribution plan

The package: the kernel benchmark (SpectroBench), the IEEE whitepaper with
documented results, the PINN spectral-bias demo, and the explainer animation.
The goal is expertise-reference material across channels, sequenced by reach.

## Channels, in priority order (reach-driven)

1. LinkedIn (primary reach, ~1800 followers, the strongest channel). Post the
   MP4 as a NATIVE upload (LinkedIn suppresses external-link posts and favors
   native video). Content-first, link non-dominant (in the comments), per the
   house rule. Draft caption below.
2. Zenodo (citable DOI, credibility). Archive a GitHub release for a permanent
   DOI; Felix already uses Zenodo. Enable the GitHub-Zenodo integration, then
   cut a tagged release (v0.1.0) and Zenodo mints the DOI automatically.
   CITATION.cff is in place.
3. HuggingFace Space (interactive demo, ML-community reach). Push the `space/`
   folder as a Gradio Space; it runs the plain-vs-Fourier comparison live.
   Optionally add a short model/dataset card linking back here.
4. GitHub (code home). Tag v0.1.0, write release notes, embed the GIF (done in
   the README). This is what Zenodo archives.
5. Medium (long-form writeup, ties to the existing Medium presence). One
   content-first article walking through the intuition and the benchmark, demo
   and repo linked at the end.
6. Twitter/X (secondary, ~50 followers) and YouTube (reach problem, 3-5
   followers; videos underperform regardless of content quality). Repost the
   video for completeness, but do not rely on them for reach. YouTube can host
   an embeddable long version, not a primary channel.

## Draft LinkedIn caption (content-first, native MP4)

Most networks that model physics quietly blur out the fine detail.

Here are two physics-informed networks trained on the exact same target: a smooth
wave carrying a fast ripple. Same budget, same data, same physics loss. Watch what
happens to the ripple.

The orange network is a standard coordinate network. It locks onto the smooth part
and never recovers the fine oscillation, the well-known spectral bias. The blue one
uses a Fourier-feature encoding and snaps to the detail within a few hundred steps.

The only difference is how the input coordinates are represented. That single choice
decides whether the high-frequency structure survives. It is also why Fourier ideas
are underrated for scientific machine learning, and the starting point of a small
benchmark I built on where Fourier representations beat local ones for PDEs, and
where they do not (shocks are the honest counter-case).

Code, the interactive demo and the write-up in the comments.

## What needs Felix's accounts / actions

- Push the repo to GitHub (from SourceTree); the sandbox has no network to push.
- Create the HuggingFace Space and push `space/` (or link this repo).
- Turn on GitHub-Zenodo, cut release v0.1.0, get the DOI, add it to the README
  and CITATION.cff.
- Post the MP4 natively on LinkedIn with the caption above.
