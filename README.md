# A proof-of-concept implementation of BL-FlowSOM
BL-FlowSOM is an improved version of [FlowSOM](https://github.com/SofieVG/FlowSOM) that uses batch learning, which is available on [Sony’s Spectral Flow Analysis (SFA) - Life sciences Cloud Platform ](https://www.sonybiotechnology.com/us/instruments/sfa-cloud-platform/). This repository contains a proof-of-concept implementation of BL-FlowSOM algorithm to reproduce clustering result of it. Please note that the clustering result of BL-FlowSOM and this proof-of-concept implementation writen in Python are almost identical, although there are some minor differences that are assumed to be due to in processing systems of the programming languages.

## Overview
This proof-of-concept implementation takes csv file as input, and outputs the trained SOM vector and the classified result of input events to SOM as clustering result. Each row of the input file is one event, and each column is each parameter data of the input dataset.

## Requirements

To run this project, you need:
- Python 3.11.0 or later
- pip 25.0 or later

## Setup

1. Clone this repository:
    ```bash
    git clone https://github.com/sony/bl-flowsom_poc
    ```

2. Move to the cloned directory:
    ```bash
    cd bl-flowsom_poc/
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Data preparation
Input data file must be in csv format without headers and can be generated as follows, for example, using fcs file as input and R flowCore package:

```bash
write.table(exprs(read.FCS("FP7000_34c.fcs", truncate_max_range = FALSE))[,1:34], "data/FP7000_34c.csv", sep=",", row.names=F, col.names=F)
write.table(exprs(read.FCS("Levine_32dim.fcs", truncate_max_range = FALSE))[,5:36], "data/Levine_32dim.csv", sep=",", row.names=F, col.names=F)
write.table(exprs(read.FCS("Levine_13dim.fcs", truncate_max_range = FALSE))[,1:13], "data/Levine_13dim.csv", sep=",", row.names=F, col.names=F)
write.table(exprs(read.FCS("Samusik_01.fcs", truncate_max_range = FALSE))[,9:47], "data/Samusik_01.csv", sep=",", row.names=F, col.names=F)
write.table(exprs(read.FCS("Samusik_all.fcs", truncate_max_range = FALSE))[,9:47], "data/Samusik_all.csv", sep=",", row.names=F, col.names=F)
```
For more information on the fcs files listed above, see [Evaluation of FlowSOM and BL-FlowSOM clustering result](https://github.com/sony/bl-flowsom_eval/).

## Usage
Perform the proof-of-concept code as follows:

```bash
python ./build_batchSOM.py -d 10 -s 1 -i ./data/FP7000_34c.csv -o results/FP7000_34c_clust.csv -c results/FP7000_34c_codes.csv
```

Arguments for _build_batchSOM.py_ are as follows:
- `-h`: show help
- `-d`: witdth and height of the SOM grid
- `-s`: seed [optional]
- `-i`: events data file in csv format (input file)
- `-o`: classified result of input events to SOM in csv format (output file)
- `-c`: trained SOM vector in csv format (output file)

## License
This repository is licensed under the [Creative Commons BY-NC-SA 4.0 License](LICENSE).
