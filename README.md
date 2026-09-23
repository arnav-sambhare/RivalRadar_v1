# Rival Radar

Rival Radar tracks a startup's competitors and writes a weekly **"What Moved"** brief: a PDF of at most two pages covering what rivals published, patented and who they hired, ranked by how much each change matters to the startup.

You give it a startup's name and a sentence or two about what it does. It finds that startup's competitors in public patent and research data, then follows them week to week. A list of known rivals is optional: discovery works from the description alone, in any sector.

## Quick start

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API keys (see [API keys](#api-keys)), then run:

```bash
.venv/Scripts/python -m rivalradar run
```

It asks for the startup's name and description, and writes the brief to `output/what-moved-YYYY-MM-DD.pdf`. On macOS or Linux, the interpreter is `.venv/bin/python`.

## How it works

Each run goes through five stages. Every data source is isolated: if one fails or has no key, it's skipped with a warning and the run still produces a brief.

### 1. Collect signals for tracked rivals

| Signal | Source | Notes |
|---|---|---|
| Publications | [OpenAlex](https://openalex.org), [arXiv](https://arxiv.org) | Each rival is matched to its OpenAlex institution. arXiv has no reliable affiliation data, so its papers are found through the researchers OpenAlex places at each rival. |
| Patents | [EPO Open Patent Services](https://developers.epo.org) | Searched by applicant name, then kept only when an applicant matches the rival after normalising legal forms and country tags ("RIVAL ONE INC [US]" = "Rival One"). |
| Hires | OpenAlex affiliation histories | A researcher who joined a rival recently, having been at another organisation before. |
| Open roles | Careers pages | Optional and best-effort. Off by default. |

### 2. Discover new rivals

Discovery searches around your seed rivals, if any, and around the startup itself:

- **Topic search.** An LLM turns the description into short phrases in patent vocabulary: some for the technology, some for the product and market. Companies that appear on recent patents (EPO) and papers (OpenAlex) matching those phrases become candidates.
- **Technology × market patent classes.** The LLM also infers CPC patent classes for the technology (how the product is made or works) and for the market (what the product is and where it's used). Patents classed in both mark the startup's niche, and this finds competitors even when their patents use unfamiliar vocabulary.
- **Crowded fields.** When a search matches too many patents to be meaningful, narrower ones take over: a technology phrase combined with a market phrase, and more detailed technology classes paired with market classes.
- **Co-authorship.** Companies that repeatedly publish with the seed rivals, or with the startup itself if OpenAlex knows it.

Candidates are then narrowed down:

1. **Companies only.** Universities, research institutes, and inventors filing in their own name are dropped.
2. **Ranking.** A single patent from a narrow (niche) search is enough to qualify. Niche matches rank first. Within each group, candidates with fewer patents rank higher, because a company with many patents in a broad search is usually an incumbent.
3. **LLM relevance check.** Each candidate is shown to the LLM with the titles of the documents that found it. Clear non-competitors are dropped: generic suppliers, likely customers or partners, and people.
4. **Small-startup filter.** A candidate is excluded if it has more than `max_patents` EPO patents or, when OpenAlex knows it, more than `max_works` papers. Checking stops once `max_new_rivals` candidates have passed, so large firms don't use up the slots.

Discovered rivals are tracked from then on. Discovery only ever expands from the seeds and the startup, never from rivals it discovered itself, so the tracked set doesn't snowball.

### 3. Save state

Each run writes a JSON snapshot and updates a local SQLite database. The snapshot is written before anything else can fail, so the next run always has a baseline to compare against.

### 4. Find what moved

This run's snapshot is compared with the previous one. New papers, patents, hires and newly tracked rivals are the week's changes. The same paper found by both arXiv and OpenAlex is merged. A newly discovered rival is reported once, as newly tracked; its existing papers and patents become its baseline rather than news.

### 5. Rank and write the brief

The LLM ranks the changes against the startup's description and gives a one-line reason for each. Without an LLM, changes are ordered by type and date. The PDF has four sections: rivals moved, new publications, new patents and hiring signals. If the brief runs over the page limit, the lowest-ranked items are cut until it fits.

## Setup

Requires Python 3.10 or later. Install dependencies into a virtual environment as shown in [Quick start](#quick-start).

### API keys

Keys come from environment variables only, never from `config.yaml`. The easiest way to set them is a `.env` file in the repository folder: copy `.env.example` to `.env` and fill it in. `.env` is gitignored and is found wherever you run the command from. A variable set in your real environment overrides the file.

| Variable | Used for | Without it | Get it |
|---|---|---|---|
| `GROQ_API_KEY` | Search phrases and classes for discovery, the relevance check, and ranking the brief | Discovery uses only a short description or `--target-keywords`, and the brief is ordered by type and date | [console.groq.com](https://console.groq.com) |
| `EPO_OPS_KEY`, `EPO_OPS_SECRET` | Patents, patent-based discovery and the size filter | All three are skipped | Free account at [developers.epo.org](https://developers.epo.org) |
| `OPENALEX_API_KEY` | Optional: avoids OpenAlex's rate limits on anonymous searches | OpenAlex still works, but can be slower | [openalex.org](https://openalex.org/rest-api) |

## Running it

```bash
.venv/Scripts/python -m rivalradar run
```

The startup isn't stored in the config. Each run asks for it:

```
Target company name: Acme Edge AI
What does it do? (1-2 sentences, used to rank relevance): Small open-weight language models for on-device use.
```

Discovery and ranking are judged against that description, so be specific about the product and its market. To skip the prompt, pass the details as flags:

```bash
.venv/Scripts/python -m rivalradar run --target-name "Acme Edge AI" --target-description "Small open-weight language models for on-device use."
```

| Option | Effect |
|---|---|
| `--target-name`, `--target-description` | The startup, instead of the prompt. Required for unattended runs. |
| `--target-keywords "a, b"` | Discovery's search phrases, instead of letting the LLM derive them. The LLM still infers the patent classes. |
| `--config PATH` | Use another config file. Relative paths in it resolve against its folder. |
| `--no-discover` | Track only the seed rivals and rivals discovered earlier. |
| `-v` | Log progress. |

After a run, a summary shows the number of rivals tracked and discovered, the search phrases and class searches used (flagging any phrase that matched nothing), what the relevance check dropped, and where the brief was saved.

A run with discovery makes a few dozen patent requests, spaced out to respect EPO's limits, so it typically takes 5–15 minutes.

### Schedule it weekly

Scheduling is left to the operating system. An unattended run has no one to answer the prompt, so it must pass the startup as flags.

With cron (Mondays at 07:00):

```
0 7 * * 1 cd /path/to/RivalRadar && .venv/bin/python -m rivalradar run --target-name "Acme Edge AI" --target-description "Small open-weight language models for on-device use."
```

On Windows, create a weekly Task Scheduler task that runs `.venv\Scripts\python.exe` with the arguments `-m rivalradar run --target-name "..." --target-description "..."`, and "Start in" set to the repository folder.

## Configuration

`config.yaml` holds everything except secrets and the startup. The settings you're most likely to change:

| Setting | Default | What it does |
|---|---|---|
| `contact_email` | empty | Your email, sent to OpenAlex and EPO. Both treat identified clients better, and EPO is less likely to block them. Recommended. |
| `seed_rivals` | none | Competitors you already know, each with a `name` and optional `openalex_institution_id`. Add any competitor discovery can't find, such as one with no patents or papers yet. |
| `small_startup_filter.max_patents` | 20 | Organisations with more EPO patents than this are too big to track. |
| `small_startup_filter.max_works` | 500 | Organisations with more OpenAlex papers than this are too big. This catches incumbents whose patent name differs from their OpenAlex name. |
| `discovery.max_new_rivals` | 25 | The most rivals a single run can add. |
| `discovery.max_candidates` | 40 | Candidates checked by the relevance check and the size filter. |
| `discovery.patent_offices` | EP, US, WO, GB, FR, DE | Patent offices discovery searches. Without a restriction, over 90% of topical patents are Chinese national filings. Add `"CN"`, or use `[]` for all offices, if competitors file mainly in China. |
| `discovery.niche_query_max_hits` | 50 | A technology search matching at most this many patents is niche, and one patent from it is enough to qualify. |
| `discovery.min_cooccurrence` | 2 | Matches needed from non-niche searches. |
| `discovery.relevance_check` | true | Whether the LLM screens candidates before the size filter. |
| `lookback_days` | 7 | How far back each run collects. The diff only reports what the previous snapshot didn't have, so a longer window is safe and catches papers OpenAlex indexes late. |
| `llm.model` | `qwen/qwen3.8-27b` | The Groq model. |
| `llm.max_items` | 12 | Changes kept in the brief. |
| `output.max_pages` | 2 | Page limit for the brief. |

Other settings, all commented in `config.yaml`:
- `epo.min_interval`, `epo.retries`: EPO request pacing and retries.
- `llm.min_interval`, `llm.max_output_tokens`, `llm.output_tokens_per_minute`, `llm.screen_batch_tokens`: Groq pacing and request sizes.
- `discovery.topic_years`, `discovery.cpc_max_hits`, `discovery.cpc_groups`, `discovery.institution_types`: discovery tuning.
- `arxiv.*`, `hiring.max_join_age_years`, `sources.*`, `storage.*`: other sources and paths.

### Staying within API limits

Both external APIs block clients that behave like robots, so Rival Radar paces itself:

- **EPO** blocks keys that send bursts of searches. Every request waits `epo.min_interval` seconds (10 by default), and longer when EPO reports it's busy. Each run reuses one connection and one access token. If EPO flags the key anyway, the run makes no further EPO calls.
- **Groq** limits tokens per minute (input and output separately) and requests per day. Calls are spaced out. Each reply is capped at `llm.max_output_tokens`, and the output-token limit (`llm.output_tokens_per_minute`) isn't reported in Groq's response headers, so set it to your plan's figure if it differs from 1,000. The client waits for the per-minute budgets to reset, and stops calling once the day's quota is used up, falling back to non-LLM behaviour.

## State

Everything below is local and gitignored:

| Path | Contents |
|---|---|
| `snapshots/snapshot-*.json` | One per run. The latest is the next run's baseline. |
| `rivalradar.sqlite` | Every record collected, plus discovered rivals, which stay tracked. |
| `output/` | The PDF briefs. |

The first run has no baseline, so everything counts as new, and the brief says so.

Tracked rivals depend on the startup you entered. **To follow a different startup, start from fresh state:** delete `snapshots/` and `rivalradar.sqlite`, or use a separate config file with its own `storage` paths. Otherwise rivals discovered for one startup stay tracked for the next.

## Project layout

```
rivalradar/
  cli.py          command line and the weekly pipeline
  config.py       config.yaml and secrets from the environment / .env
  models.py       Rival, Publication, Patent, Affiliation, SignalBundle
  sources/        one module per source: openalex, arxiv, epo, careers,
                  plus registry.py (runs them) and base.py (HTTP, fail-soft)
  discovery.py    rival discovery: search terms, candidates, relevance check
  filters.py      small-startup size filter
  names.py        organisation-name normalisation and matching
  llm.py          Groq calls within rate limits
  diff.py         week-over-week changes
  relevance.py    LLM ranking of changes
  report.py       the PDF brief
  db.py           SQLite storage
  snapshot.py     JSON snapshots
tests/            offline test suite
```

## Tests

```bash
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

```bash
.venv/Scripts/python -m pytest
```

The tests run offline. External APIs are replaced with fakes; snapshots, the database, the diff, ranking and the PDF run for real, including multi-week pipeline runs.

## Known limitations

- **Discovery only sees companies with public patents or papers.** A startup that hasn't filed or published yet, or files under a different name, can't be discovered. Add it to `seed_rivals`.
- **Discovery depends on the LLM's search terms.** The phrases and patent classes it infers vary in quality, and the run summary shows which were used and which matched nothing. If they're off, set the phrases yourself with `--target-keywords`.
- **The size filter is a proxy.** Patent and paper counts approximate company size. A well-funded startup with a large patent portfolio can be excluded, so raise `max_patents` if that matters.
- **Hire signals are noisy.** OpenAlex sometimes merges different people who share a name, producing wrong prior employers or join dates. The LLM is told to mark down implausible hires, and the brief carries a warning. Verify before acting on one.
- **Without EPO keys, discovery can't check size.** Companies are then kept whatever their size, and the run summary lists them.
- **Name matching is prefix-based.** "Rival One" also matches an applicant called "Rival One-Off Widgets".
