<p align="center">
  <img src="Textream/Textream/Assets.xcassets/AppIcon.appiconset/icon_256x256.png" width="96" height="96" alt="Textream icon">
</p>

<h1 align="center">Textream · Semantic Auto-Advance (fork)</h1>

<p align="center">
  <strong>A macOS teleprompter that keeps following you even when you go <em>off-script</em> — it advances by <em>meaning</em>, not exact words. Fully on-device, no API key, no network.</strong>
</p>

---

> ### Provenance & attribution — please read first
>
> This repository is a **fork** of **[f/textream](https://github.com/f/textream)** — *Textream*, a free, 3k★ open-source macOS teleprompter created by **Fatih Kadir Akın** ([@f](https://github.com/f) · [textream.fka.dev](https://textream.fka.dev/)). **The app itself is the upstream author's work.** GitHub preserves the "forked from f/textream" link.
>
> The original author's full README (features, download, Homebrew, how-it-works) is kept in-repo at **[`docs/ORIGINAL-README.upstream.md`](docs/ORIGINAL-README.upstream.md)** and lives upstream at **[github.com/f/textream](https://github.com/f/textream)**.
>
> **What this fork adds — my contribution, and the point of this portfolio repo:** a local **semantic paraphrase auto-advance** layer so the prompter keeps up when you rephrase the script in your own words, not only when you read it verbatim.

## The problem this fork solves

Upstream Textream tracks your script by matching the speech transcript **character/word by character/word**. Verbatim reading works beautifully — but presenters paraphrase. The instant you say a line's *meaning* in different words, literal matching loses you and the prompter stalls.

This fork adds a **local semantic layer in parallel** to the existing literal matcher: when literal matching is unsure, it compares *what you actually said* against upcoming script segments by **meaning** (on-device sentence embeddings + cosine similarity) and advances when they're semantically close — strongly biased toward "wait" over "jump to the wrong place."

## What I added

| Area | File | Summary |
|---|---|---|
| Local semantic engine | `Textream/Textream/SemanticMatcher.swift` *(new)* | Apple **`NLContextualEmbedding`** (512-dim, transformer, on-device), mean-pooled sentence vectors + cosine similarity. Offline, no API key, no network. |
| Matching-core integration | `Textream/Textream/SpeechRecognizer.swift` *(+247 / −20)* | Semantic layer wired **in parallel** to the original literal matcher (literal high-confidence is never overridden), behind a confirmation gate and several anti-false-advance guards. |

### Design highlights

- **Hybrid, non-regressive** — the original char/word matcher is fully preserved; semantics only assist when literal matching doesn't commit. Verbatim reading behaves exactly as upstream.
- **Anti-false-advance multi-gate** — conservative similarity threshold, the existing 2-of-3 confirmation gate, forward-only candidate window, stale-result drop, no small-step bypass for semantic jumps.
- **Pace anchor** — the pointer can never run ahead of how much you've actually spoken (projected into script units), so it won't bury lines you haven't said yet. Strictly biased toward lagging over leading.
- **Bounded recent-speech window** — robust to Chinese continuous dictation lacking sentence punctuation (prevents the semantic vector from being diluted by the whole cumulative transcript).
- **Graceful offline degradation** — if the embedding model isn't ready it silently falls back to pure literal matching; never blocks, never errors.

### Local engine benchmark

Measured by an offline regression harness (Apple `NLContextualEmbedding`, mechanism-faithful metric = forward-candidate argmax at the frozen operating threshold). Real harness output, not estimates:

- **Hit rate: 91.1%** (paraphrased lines correctly advanced)
- **False-advance: 5.6%**
- **~12 ms** per sentence-pair embed + cosine, on-device

Tuning parameters are current candidate values pending a larger grid-search pass.

## Build & run

Requires **macOS 15+** and **Xcode**.

```bash
git clone https://github.com/kkunkunya/textream.git
cd textream
bash build.sh          # builds a universal (arm64 + x86_64) Textream.app
open build/release/universal/Textream.app
```

In the app: paste your script, grant **Microphone** + **Speech Recognition**, set **Speech Language** to your language, start, and speak — paraphrase freely; the prompter follows your meaning. For the original app's full feature list, modes, and Homebrew install, see [`docs/ORIGINAL-README.upstream.md`](docs/ORIGINAL-README.upstream.md).

## Licensing & attribution

The upstream repository **[f/textream](https://github.com/f/textream)** publishes **no license** — by default that means *all rights reserved* by the original author. Therefore:

- This is a **GitHub fork** (provenance link preserved). It is **not** re-published as original work, and **no relicensing** is claimed or applied over the upstream code.
- The semantic auto-advance code I added (`SemanticMatcher.swift` and the semantic-integration portions of `SpeechRecognizer.swift`) is my own contribution; to reuse *that* part, please open an issue.
- For any use of the app as a whole, refer to the upstream author **Fatih Kadir Akın** / [textream.fka.dev](https://textream.fka.dev/).

## Credits

- **Original app:** [Textream](https://github.com/f/textream) by **Fatih Kadir Akın** ([@f](https://github.com/f)).
- **This fork's addition:** local semantic paraphrase auto-advance — by [@kkunkunya](https://github.com/kkunkunya).
