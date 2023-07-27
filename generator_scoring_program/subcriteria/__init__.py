import sys 

sys.path.append("..")   
from config import USE_OUR_BASELINE_REVIEWER 

if USE_OUR_BASELINE_REVIEWER:
    from .clarity import *
    from .responsibility import *
    from .confidence import *
    from .contribution import *
    from .soundness import *
    from .relevance import *

from .utils import *