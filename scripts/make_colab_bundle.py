"""Pack everything the Colab notebook needs into colab_input.zip (upload it to MyDrive/p1_tsfm/)."""
import zipfile

from cpd import ROOT

OUT = ROOT / "colab_input.zip"
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in ("configs", "src/cpd", "data/processed"):
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                z.write(p, p.relative_to(ROOT).as_posix())
    z.write(ROOT / "requirements-colab.txt", "requirements-colab.txt")
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(zipfile.ZipFile(OUT).namelist())} files)")
