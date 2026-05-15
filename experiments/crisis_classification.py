"""Pre-registered endogenous-vs-exogenous classification of the 17 crises.

CRITERION (committed before observing gap data on a per-crisis basis):
    A crisis is EXOGENOUS if there is a single dominant external trigger event
    with an identifiable date — a news headline, institutional failure, policy
    announcement, or natural event — whose occurrence on its own would suffice
    to explain the onset of the crisis.

    Otherwise, the crisis is ENDOGENOUS — onset emerges from market-internal
    dynamics (positional unwinding, microstructural cascades, feedback loops)
    without a single external trigger.

The classification below is written from publicly available sources (Wikipedia,
Bloomberg coverage, Bouchaud's work on endogenous vs exogenous crashes) and
fixed before the formal test against the spectral-gap data.

Key references for the framework:
    - Sornette (2002) "Endogenous versus exogenous origins of financial rallies and crashes"
    - Filimonov & Sornette (2015) "Quantifying reflexivity in financial markets"
    - Bouchaud (2024) "Self-organized criticality in economics and finance" (SSRN)

Borderline cases are flagged explicitly with `borderline=True`. The default
test uses the strict classification; a sensitivity analysis can drop the
borderlines.
"""

from __future__ import annotations

# id: classification, trigger_or_reasoning, borderline_flag
CRISIS_CLASSIFICATION = {
    "1997_asia": {
        "class": "exogenous",
        "reasoning": "Thai baht devaluation (2 July 1997) was the trigger; Asian sovereign defaults followed mechanically.",
        "borderline": False,
    },
    "1998_ltcm": {
        "class": "exogenous",
        "reasoning": "Russian sovereign default (17 August 1998) directly triggered LTCM's collapse.",
        "borderline": False,
    },
    "2000_dotcom": {
        "class": "endogenous",
        "reasoning": "No single trigger; gradual deflation of dot-com valuations from 2000-Q1 onward driven by exhaustion of speculative demand.",
        "borderline": False,
    },
    "2001_911": {
        "class": "exogenous",
        "reasoning": "11 September 2001 attacks; market closed Sep 11-17.",
        "borderline": False,
    },
    "2007_quant": {
        "class": "endogenous",
        "reasoning": "Statistical-arbitrage and equity market-neutral funds liquidated in coordinated fashion (Aug 7-9 2007); no single external news event.",
        "borderline": False,
    },
    "2008_gfc": {
        "class": "exogenous",
        "reasoning": "Lehman Brothers bankruptcy 15 September 2008 was the dominant trigger; the broader crisis was structural but the panic onset was a single discrete event.",
        "borderline": True,  # the structural buildup was endogenous; only the panic phase is exogenous
    },
    "2010_flash": {
        "class": "endogenous",
        "reasoning": "6 May 2010 flash crash driven by HFT cascade; CFTC/SEC report attributes onset to a single sell algo but the cascade itself was microstructural.",
        "borderline": False,
    },
    "2011_euro": {
        "class": "exogenous",
        "reasoning": "Sequential sovereign-debt events (Greek/PIIGS bond yields, S&P US downgrade Aug 5 2011); each triggered by an external announcement.",
        "borderline": False,
    },
    "2013_taper": {
        "class": "exogenous",
        "reasoning": "Bernanke testimony 22 May 2013 introducing tapering language was the trigger.",
        "borderline": False,
    },
    "2015_china": {
        "class": "exogenous",
        "reasoning": "Yuan devaluation 11 August 2015 + Shanghai Composite collapse triggered by PBoC policy actions.",
        "borderline": False,
    },
    "2016_brexit": {
        "class": "exogenous",
        "reasoning": "Brexit referendum result 24 June 2016.",
        "borderline": False,
    },
    "2018_volmageddon": {
        "class": "endogenous",
        "reasoning": "5 February 2018 XIV/SVXY mechanical unwind; positional cascade with no external news trigger.",
        "borderline": False,
    },
    "2018_q4": {
        "class": "endogenous",
        "reasoning": "Broad-based Q4 selloff; no single triggering event. Often attributed to year-end positioning + Fed-cycle anxiety but no single news event explains the onset.",
        "borderline": True,  # Fed signaling could be argued as exogenous
    },
    "2019_repo": {
        "class": "exogenous",
        "reasoning": "Repo rate spike 17 September 2019, attributed to corporate tax payment + Treasury issuance; specific date and mechanism.",
        "borderline": False,
    },
    "2020_covid": {
        "class": "exogenous",
        "reasoning": "COVID-19 pandemic; WHO declared pandemic 11 March 2020. The largest exogenous shock in the dataset.",
        "borderline": False,
    },
    "2021_meme": {
        "class": "endogenous",
        "reasoning": "GameStop short squeeze (late January 2021) was driven by retail coordination and dealer hedging cascades; no external trigger.",
        "borderline": False,
    },
    "2022_rates": {
        "class": "endogenous",
        "reasoning": "Cumulative reaction to Fed tightening cycle; gradual repricing with no single onset event. (Strict reading of criterion: no single dominant trigger.)",
        "borderline": True,  # Fed's first rate hike could be argued as exogenous
    },
    "2023_svb": {
        "class": "exogenous",
        "reasoning": "Silicon Valley Bank collapse 10 March 2023; specific institutional failure.",
        "borderline": False,
    },
    "2024_carry": {
        "class": "exogenous",
        "reasoning": "Bank of Japan rate hike 31 July 2024 triggered yen carry-trade unwind on 5 August 2024.",
        "borderline": True,  # carry-trade unwind cascade is endogenous; BoJ hike was the trigger
    },
}


def get_class(crisis_id: str) -> str | None:
    entry = CRISIS_CLASSIFICATION.get(crisis_id)
    return entry["class"] if entry is not None else None


def get_classification_summary() -> dict:
    n_endo = sum(1 for v in CRISIS_CLASSIFICATION.values() if v["class"] == "endogenous")
    n_exo = sum(1 for v in CRISIS_CLASSIFICATION.values() if v["class"] == "exogenous")
    n_borderline = sum(1 for v in CRISIS_CLASSIFICATION.values() if v["borderline"])
    return {
        "n_total": len(CRISIS_CLASSIFICATION),
        "n_endogenous": n_endo,
        "n_exogenous": n_exo,
        "n_borderline": n_borderline,
        "endogenous": [k for k, v in CRISIS_CLASSIFICATION.items() if v["class"] == "endogenous"],
        "exogenous": [k for k, v in CRISIS_CLASSIFICATION.items() if v["class"] == "exogenous"],
        "borderline": [k for k, v in CRISIS_CLASSIFICATION.items() if v["borderline"]],
    }


if __name__ == "__main__":
    import json
    s = get_classification_summary()
    print(json.dumps(s, indent=2))
    print()
    print("Per-crisis classification (with reasoning):")
    for crisis_id, entry in CRISIS_CLASSIFICATION.items():
        flag = " [BORDERLINE]" if entry["borderline"] else ""
        print(f"  {crisis_id:<22} {entry['class']:<11}{flag}")
        print(f"    {entry['reasoning']}")
