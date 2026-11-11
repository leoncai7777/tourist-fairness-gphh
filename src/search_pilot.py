"""
Pilot of the search (Step 4f, F1)
=================================

Section 6 of search_design.md. Whole runs, end to end: initialisation, every
generation with monitoring and checkpoints, the end of the run on all
validation days, and the files written. Training and validation days only.
Pilot runs use run numbers from 900, so the seeds and schedules of the formal
runs (0 to 29) are never used here. Pilot results are not results of the
study: they only fix N, G, K and the normalisation range.

  python3 src/search_pilot.py short   # every method, two short runs each, all at once
  python3 src/search_pilot.py full    # GP-fixed, GP-prop, GP-adaptive at N = 200, G = 100, K = 5, at once

Each run is its own process (one level of parallelism); it writes its folder
under outputs/search/pilot/<phase>/ and a pilot.json with wall time, peak
memory and output sizes. Writes outputs/search/pilot/<phase>/record.md.
"""

import json
import resource
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("outputs/search/pilot")
PHASES = {
    "short": {"methods": ["GP-single", "GP-fixed", "GP-prop", "GP-adaptive"], "runs": [900, 901],
              "settings": {"population": 40, "generations": 20, "crossover": 0.8, "mutation": 0.15,
                           "monitor every": 5, "validation replicates": 1}},
    "full": {"methods": ["GP-fixed", "GP-prop", "GP-adaptive"], "runs": [910],
             "settings": {"population": 200, "generations": 100, "crossover": 0.8, "mutation": 0.15,
                          "monitor every": 10, "validation replicates": 5}},
    # the code with the settings wiring and the evaluation cache, end to end at full scale
    "current": {"methods": ["GP-adaptive"], "runs": [911],
                "settings": {"population": 200, "generations": 100, "crossover": 0.8, "mutation": 0.15,
                             "monitor every": 10, "validation replicates": 5}},
}


def worker(method, run_number, phase):
    """One whole run in this process."""
    sys.path.insert(0, str(Path(__file__).parent))
    from search import Run, finish, hypervolume, normalise
    settings = PHASES[phase]["settings"]
    folder = OUT / phase / f"{method}_{run_number}"
    started = time.time()
    run = Run(method, run_number, settings, folder=folder / "checkpoint")
    run.evolve()
    searched = time.time()
    out = finish(run, folder / "results")
    finished = time.time()
    monitored = [{"generation": m["generation"], "members": len(m["front"]),
                  "hypervolume": hypervolume(normalise(m["front"]))[0]} for m in run.monitor_log]
    files = {p.name: p.stat().st_size for p in (folder / "results").iterdir()}
    from search import code_version
    report = {"method": method, "run": run_number, "settings": settings, "code version": code_version(),
              "search seconds": searched - started, "end seconds": finished - searched,
              "peak memory MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,  # bytes on macOS
              "generations": run.log, "monitoring": monitored, "summary": out["summary"],
              "front": out["front"][["text", "copies", "mean R_v", "G_hat", "knee"]].to_dict("records"),
              "files": files}
    (folder / "pilot.json").write_text(json.dumps(report, indent=1, default=str))


def launch(phase):
    """Every run of the phase at once, one process each; then the record."""
    spec = PHASES[phase]
    started = time.time()
    processes = [subprocess.Popen([sys.executable, __file__, "worker", method, str(number), phase])
                 for method in spec["methods"] for number in spec["runs"]]
    codes = [p.wait() for p in processes]
    if any(codes):
        raise SystemExit(f"a pilot run failed: exit codes {codes}")
    write_record(phase, time.time() - started)


def write_record(phase, wall):
    reports = [json.loads(p.read_text()) for p in sorted((OUT / phase).glob("*/pilot.json"))]
    s = PHASES[phase]["settings"]
    lines = [f"# Pilot of the search, phase {phase}", "",
             f"Settings {s}. {len(reports)} runs at once, one process each; {wall / 60:.0f} minutes in all. "
             f"Training and validation days only; run numbers from 900. Not results of the study.", "",
             "## Each run", "",
             "| Method | Run | Search (min) | End of run (min) | Peak memory (MB) | Seconds per training evaluation "
             "| Validation hypervolume | Front members | Front rules | Output (MB) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        evaluations = 2 * s["population"] * s["generations"]
        lines.append(f"| {r['method']} | {r['run']} | {r['search seconds'] / 60:.1f} | {r['end seconds'] / 60:.1f} | "
                     f"{r['peak memory MB']:.0f} | {sum(g['seconds'] for g in r['generations']) / evaluations:.3f} | "
                     f"{r['summary']['hypervolume']:.3f} | {r['summary']['front members']} | "
                     f"{r['summary']['front rules']} | {sum(r['files'].values()) / 1e6:.1f} |")
    lines += ["", "Hypervolumes use the provisional range of search.py, only to watch the search; the pilot then "
              "fixes the range.", "", "## Monitoring and search activity, by generation", ""]
    for r in reports:
        lines.append(f"### {r['method']}, run {r['run']}")
        lines.append("")
        lines.append("| Generation | Monitored front members | Monitored hypervolume (7 validation days) | Unique share "
                     "| Largest share | Crossover: new texts | Mutation: new texts | Mean nodes |")
        lines.append("|---|---|---|---|---|---|---|---|")
        by_generation = {m["generation"]: m for m in r["monitoring"]}
        for g in r["generations"]:
            m = by_generation.get(g["generation"])
            members = m["members"] if m else ""
            volume = f"{m['hypervolume']:.3f}" if m else ""
            crossover = g["crossover new texts"] / g["crossover children"] if g["crossover children"] else float("nan")
            mutation = g["mutation new texts"] / g["mutation children"] if g["mutation children"] else float("nan")
            lines.append(f"| {g['generation']} | {members} | {volume} | {g['unique share']:.2f} | "
                         f"{g['largest share']:.2f} | {crossover:.2f} | {mutation:.2f} | {g['mean nodes']:.1f} |")
        lines.append("")
    monitored = sum(m["members"] for r in reports for m in r["monitoring"])
    lines += [f"Monitoring evaluated {monitored * 7:,} rule-days in all (front members, copies included), "
              f"counted apart from the distinct front rules re-run at the end.", ""]
    (OUT / phase / "record.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        launch(sys.argv[1])
