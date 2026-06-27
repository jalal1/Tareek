# Config

Starter configuration files for Tareek, organized by country/region under
[`USA/`](USA/). Each city folder holds a config we have actually used in the
past as a starting point for that area.

| City | File |
|------|------|
| [Birmingham, AL](USA/Birmingham_AL/config_birmingham.json) | `config_birmingham.json` |
| [Chicago](USA/Chicago/config_chicago.json) | `config_chicago.json` |
| [Washington, DC](USA/DC/config_washington_dc.json) | `config_washington_dc.json` |
| [Los Angeles](USA/LA/config_la.json) | `config_la.json` |
| [New York](USA/NewYork/config_newyork.json) | `config_newyork.json` |
| [Twin Cities (Minneapolis–St. Paul)](USA/TwinCities/config_twincities.json) | `config_twincities.json` |

## How to use these

These are **starting points**, not turn-key configs. Before running one:

- **Read it carefully.** Paths, region/county selections, scaling factors, and
  iteration counts are tuned for the run that produced them — they will not all
  be correct for your machine or your goal.
- **Adjust paths and inputs** to match your local setup.
- **Change values deliberately.** Understand what a parameter does before
  editing it; the defaults reflect choices made for a specific city and run.

## Best results so far

The most up-to-date, best-performing configurations and their documented results
live in the [`examples/`](../examples/) folder. Each example records a real run
for a region — what was run, the configuration used, and the results. We will
keep adding experiments from more cities across the USA there, so check it for
the latest recommended setup before tuning your own.

The Twin Cities config here (`USA/TwinCities/config_twincities.json`) was copied from
its example so you can start from a known-good run; see
[`examples/twin-cities-jalal-20260623/`](../examples/twin-cities-jalal-20260623/)
for the accompanying results and write-up.
