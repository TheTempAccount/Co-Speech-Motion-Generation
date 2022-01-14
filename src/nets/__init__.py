from .freeMo_Graph import TrainWrapper as freeMo_Graph_Generator
from .freeMo_Graph_v2 import TrainWrapper as freeMo_Graph_Generator_v2
from .freeMo import TrainWrapper as freeMo_Generator
from .freeMo_old import TrainWrapper as freeMo_Generator_old

from .speech2gesture import Generator as S2G_Generator
from .trimodal_context import Generator as TriCon_Generator
from .audio2body import Generator as A2B_Generator
from .base import TrainWrapperBaseClass
