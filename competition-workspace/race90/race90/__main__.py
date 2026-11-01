import time

started = time.time()
from .cli import main

raise SystemExit(main(start_time=started))
