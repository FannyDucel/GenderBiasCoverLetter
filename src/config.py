from pathlib import Path

ROOT_LOC: Path = Path("..") if str(Path().cwd()).split("/")[-1] == "src" else Path(".")
LOCATIONS = {
    "root": ROOT_LOC,
}

# Make sure all these dirs exist
for loc in LOCATIONS.values():
    loc.mkdir(exist_ok=True, parents=True)
