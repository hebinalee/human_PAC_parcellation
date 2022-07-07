# human_PAC_parcellation
## About The Project
:large_blue_diamond: This is code for the paper **"In-vivo data-driven parcellation of Heschl’s gyrus using structural connectivity"** (Published)<br />
:large_blue_diamond: **Paper link:** https://www.nature.com/articles/s41598-022-15083-z<br />
:large_blue_diamond: If you use this code, please cite the article.<br /><br />

✔ This study is divided into following 4 steps:<br />
　　　**1)** Structural connectome-based PAC parcellation<br />
　　　**2)** Characterization of subregions in terms of functional connectome pattern<br />
　　　**3)** Identification of functional hierarchy of subregions via gradient analysis<br />
　　　**4)** Anatomical analysis with myelin density & cortical thickness<br /><br />

## Directory Structure
```bash
├── 1.parcellation
│   ├── README.md
│   ├── find_optimal_k.py
│   └── PAC_clustering.py
├── 2.FC_characterization
│   ├── README.md
│   ├── compute_meanFC.py
│   └── statistical_test_FC.py
├── 3.gradient
│   ├── README.md
│   ├── gradient_analysis.py
│   └── statistical_test_FG.py
├── 4.anatomical_analysis
│   ├── README.md
│   ├── myelin_thickness.py
└── README.md
```
<br />

## License
:pushpin: **copyrightⓒ 2021 All rights reserved by Hyebin Lee and Kyoungseob Byeon**
