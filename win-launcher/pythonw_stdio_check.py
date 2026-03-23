import sys
from pathlib import Path
out = Path(r"C:\Users\TheunisvanNiekerk\Code\Presentation\RL\pythonw_stdio_check.txt")
out.write_text(f"stdout={sys.stdout!r}\nstderr={sys.stderr!r}\nstdin={sys.stdin!r}\n")
