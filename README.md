This repository copy (Python 3.6) is for review of the paper titled "Interconnectedness enhances network resilience of
multimodal public transportation systems for Safe-to-Fail urban mobility"

Authors: Zizhen Xu, Shauhrat S. Chopra*

Email: S.S.C. <sschopra@cityu.edu.hk>, X.Z. <zizhenxu2@cityu.edu.hk>




**Note: analysis in the paper only used part of the functions in the provided resilience analysis framework.**

**Software & system:**\
PyCharm PyCharm 2023.1.2 (Community Edition)\
Build #PC-231.9011.38, built on May 17, 2023\
Runtime version: 17.0.6+10-b829.9 x86_64\
VM: OpenJDK 64-Bit Server VM by JetBrains s.r.o.\
macOS 13.4\
GC: G1 Young Generation, G1 Old Generation\
Memory: 8192M\
Cores: 12\
Metal Rendering is ON\
Registry:\
debugger.new.tool.window.layout=true\
ide.experimental.ui=true\
ide.balloon.shadow.size=0

Non-Bundled Plugins:\
com.andrey4623.rainbowcsv (2.0.1)\
net.seesharpsoft.intellij.plugins.csv (2.18.2)\
String Manipulation (9.2.0)\
com.firsttimeinforever.intellij.pdf.viewer.intellij-pdf-viewer (0.11.1)\
Key Promoter X (2023.1.0)

**Installation:**\
https://www.jetbrains.com/pycharm/download/#section=mac

**Essential modules:**\
networkx 2.5.1, numpy 1.17.2, pandas 0.25.1, csv, random, itertools.

**Supporting modules:**\
matplotlib 3.3.4, multiprocessing, math, tqdm 4.41.1, collections, copy.

**Expect time to install the software and modules:**\
30 minutes

**Expect time to run analysis:**\
300 hours for reproduction (depending on the number of repeated tests; reduce it for
demo run)

**Instruction:**\
please run the python files in the following table sequentially

| Description / Paper section                                                            | Python file                                             |  
|----------------------------------------------------------------------------------------|---------------------------------------------------------|
| **Section 2.1 Characteristics of subsystems**                                          |                                                         | 
| -- Analysis of subsystems (Table 1 & Figure 4)                                         | `mptn_analyze_individual_unweighted_network.py`         |         
| -- Null model benchmark                                                                | `benchmark_ER_analyze_individual_unweighted_network.py` |          
|                                                                                        |                                                         |
| **Section 2.2 Change in characteristics during one-by-one integration**                |                                                         |              
| -- One-by-one subsystem integration (Table 2, Figure 4 & Figure 5)                     | `mptn_analyze_expanding_unweighted_network.py`          |  
| -- Null model benchmark                                                                | `benchmark_ER_expanding_unweighted_network.py`          |           
|                                                                                        |                                                         |
| **Section 2.3 Interconnectedness and network robustness of MPTN**                      |                                                         |
| -- Relationship between $D_{IMT}$ and robustness (Figure 3c)                           | `mptn_optimize_intermodal_distance.py`                  |
| -- Plot curves in Figure 3a (example data has been provided)                           | `results_plotting_tool.py`                              |
| -- Calculate Z-scores (Figure 3b & Table 2, example data has been provided)            | `results_processing_Z_scores.py`                        |
|                                                                                        |                                                         |
| **Section 2.4 Network interoperability of MPTN** (analysis performed with Section 2.2) |                                                         |
|                                                                                        |                                                         |

**Other code (don't need to run):**

| Description / *Others                                                 | Python file                  |  
|------------------------------------------------------------------------------------------|------------------------------|
| ***GTFS model** (before converted to `networkx` DiGraph model in Space-L representation) |                              |             
| -- Import raw dataset                                                                    | `mptn_modeling_from_gtfs.py` |          
| -- Network structure                                                                     | `NetworkStructure.py`        |                
| -- Route structure                                                                       | `RouteStructure.py`          |        
| -- Trip structure                                                                        | `TripStructure.py`           |
| -- Stop structure                                                                        | `StopStructure.py`           |                 
|                                                                                          |                              |
| ***Analysis framework and supporting code**                                              |                              |          
| -- Resilience analysis framework                                                         | `xc_resilience_live.py`      |               
| -- Nothing important                                                                     | `toolbox.py`                 |                  

**Note: Expected output are provided in the results folders.**