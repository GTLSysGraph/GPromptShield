from .AllInOnePrompt import HeavyPrompt, FrontAndHead, LightPrompt
from .GPPTPrompt import GPPTPrompt
from .GPrompt import Gprompt

from .MultiGPrompt import downprompt, DGI, GraphCL, Lp
from .MultiGPrompt import AvgReadout, Discriminator, LogReg
from .MultiGPrompt import Lpprompt, DGIprompt, GraphCLprompt
from .MultiGPrompt import weighted_feature, weighted_prompt, downstreamprompt, featureprompt, GcnLayers

# add by ssh
from .RobustPrompt_I        import RobustPrompt_I
from .RobustPrompt_I_Test   import RobustPrompt_I_Test, LightPrompt
from .RobustPrompt_T        import RobustPrompt_T
from .RobustPrompt_GPF      import RobustPrompt_GPF, RobustPrompt_GPFplus

from .GPF import GPF, GPF_plus

