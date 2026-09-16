# Lets the pipeline run as `python -m cowvision ...` instead of needing an installed script.
import sys
from .cli import main
sys.exit(main())
