# Measurement database (`kpfm_pl.database`)

Builds a searchable databank of paired KPFM (`.txt`) / photoluminescence
(`.ptu`) measurements from a folder archive, so you can query and reload
measurement pairs by their metadata instead of hunting through folders.

## Archive layout it understands

```
<sample>/[<precipitation>/]<date>/<...DRx..._pair files...>
```

- **sample** – top-level folder (e.g. `CF501`, `JS249`). If the folder name
  has spaces (e.g. `"JV001 DR1 07.05.2026 noLaserPulse"`), only the first
  word is used as the sample name.
- **precipitation** – optional subfolder describing degree of precipitation
  (e.g. `"4x precipitated"`). Omitted if not present.
- **date** – a `dd.mm.yyyy`-style subfolder. If date/DR information is baked
  into a single folder name instead of separate subfolders, it is parsed out
  of that name.
- The lowest ("leaf") folder that directly contains `.txt`/`.ptu` files holds
  the actual measurement pairs. `.ipynb` and any other files are ignored.

## What gets parsed out of each pair

| Field | Source | Notes |
|---|---|---|
| `dr` | either file | e.g. `"DR2"` |
| `scan` | `.txt` | `Scan1`, `Scan2`, ... |
| `kpfm_mode` | `.txt` | `"heterodyne"` (`hKPFM<amp>V`) or `"AM"` (`AM<gain>_<amp>V`, `AM<amp>V`, or `AM<gain>`) |
| `kpfm_amplitude_v` | `.txt` | modulation amplitude in V |
| `kpfm_gain` | `.txt` | recorded but not physically important |
| `pid_bandwidth_hz` | `.txt` | PID bandwidth, e.g. `25Hz`, `100mHz` |
| `wavelength_nm` | `.ptu` | defaults to `440.0` if absent from either filename |
| `power_density_w_cm2` | `.ptu` | e.g. `10W` -> interpreted as 10 W/cm² |
| `total_power_nw` | `.ptu` (or `.txt`) | e.g. `600nW`, secondary/reference only |
| `measurement_time_s` | `.ptu` | defaults to `360.0` if absent |
| `txt_path` / `ptu_path` | -- | full paths to the two files, for later loading |

`.txt`/`.ptu` pairing within a folder is done by maximizing the number of
shared tokens (DR, frequency, total power, wavelength) between candidate
files -- this correctly disambiguates folders that hold several pairs for
the same DR (e.g. distinguished only by PID frequency, or only by
power/wavelength when the `.ptu` filename doesn't carry a DR label at all).
Files sharing *no* token are never paired: they are skipped with a warning,
and ambiguous ties (several candidates with the same best score) also warn.
Data files placed directly in the scan root are supported; the root folder
name is then used as the sample name.

## Building and persisting the database

```python
from kpfm_pl import build_database, save_json, save_csv, load_json

entries = build_database(r"C:\Users\rolfs\Desktop\Claude correlate")
save_json(entries, r"C:\Users\rolfs\Desktop\Claude correlate\database.json")
save_csv(entries, r"C:\Users\rolfs\Desktop\Claude correlate\database.csv")   # for Excel

# later, in a new session:
entries = load_json(r"C:\Users\rolfs\Desktop\Claude correlate\database.json")
```

Rebuilding from scratch (`build_database`) is the simplest way to keep the
database in sync if you add/rename whole folders of raw data.

## Querying / navigating the database

`entries` is just a plain Python `list[Entry]`, so any list operation works.
Two convenience helpers are provided for the common cases.

### Select all pairs matching some criteria

```python
from kpfm_pl import filter_entries

# all pairs recorded at 532 nm
green = filter_entries(entries, wavelength_nm=532.0)

# combine criteria (AND) -- e.g. one specific sample + DR
cf501_dr2 = filter_entries(entries, sample="CF501", dr="DR2")

# any Entry field works: sample, precipitation, date, dr, scan, kpfm_mode,
# kpfm_amplitude_v, kpfm_gain, pid_bandwidth_hz, wavelength_nm,
# power_density_w_cm2, total_power_nw, measurement_time_s, txt_path, ptu_path
heterodyne_25hz = filter_entries(entries, kpfm_mode="heterodyne", pid_bandwidth_hz=25.0)
```

For anything beyond plain equality (ranges, sorting, etc.), use a normal list
comprehension -- `Entry` fields are just attributes:

```python
high_power = [e for e in entries if (e.power_density_w_cm2 or 0) > 10]
by_power = sorted(entries, key=lambda e: e.total_power_nw or 0)
```

### Select one specific pair

```python
# first match, or None if nothing matches
entry = next(iter(filter_entries(entries, sample="JV001", total_power_nw=1200.0, wavelength_nm=440.0)), None)

# or, if you already know the index / an earlier query narrowed it to one
entry = filter_entries(entries, sample="CF501", dr="DR1", date="23.01.2026")[0]
```

### Loading the actual data for a pair

```python
zurich, ptu = entry.load(**kwargs)   # -> (Zurich, PTU) instances, see README for their API, kwargs are passed into PTU(dark_counts=2.25)
```

## Editing the database

```python
from kpfm_pl import add_entry, remove_entry, save_json

# remove (needs the *full* stored path, not just the filename)
remove_entry(entries, txt_path=entry.txt_path)

# add a new pair -- parsed the same way build_database does; root lets it
# infer sample/precipitation/date from the folder location
add_entry(entries, new_txt_path, new_ptu_path, root=r"C:\Users\rolfs\Desktop\Claude correlate")

# any field can be overridden explicitly
add_entry(entries, new_txt_path, new_ptu_path, root=root, measurement_time_s=120.0)

save_json(entries, r"C:\Users\rolfs\Desktop\Claude correlate\database.json")
```

Both `add_entry` and `remove_entry` mutate `entries` in place and also return
the affected `Entry`/entries.
