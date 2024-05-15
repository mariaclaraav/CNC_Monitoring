# CNC_ConditionMonitoring

### Overview
This repository contains the first case study from my master's thesis, which focuses on enhancing feature engineering techniques for applications in Prognostics and Health Management (PHM). This case study is based on the analysis of vibration data from a CNC milling machine, as proposed by Tnani, Feil, and Diepold (2022).

## Case Study Description
### Data Collection
The dataset comprises acceleration data measured using a triaxial accelerometer (Bosch CISS Sensor) mounted on the CNC machine. The acceleration data were recorded at a sampling rate of 2 kHz in the X, Y, and Z directions. Both normal and anomalous data were collected over six different periods, each lasting six months, from October 2018 to August 2021. The data cover three different CNC milling machines (M01, M02, and M03), each performing 15 distinct processes.

### Data Characteristics
Machines: M01, M02, M03
Processes per Machine: 15
Sampling Rate: 2 kHz
Axes: X, Y, Z
Data Period: October 2018 to August 2021
Condition Labels: OK (standard condition) and NOK (anomalous condition)

### Anomalies and Failures
According to Tnani, Feil, and Diepold (2022), process failures include:

- Tool misalignment
- Presence of chips in the spindle
- Tool breakage

Post-production, a specialist inspected the pieces and annotated the process health, labeling the data accordingly.

### Analysis Approach
In this case study, traditional feature engineering techniques will be employed to prepare the data for conventional classification models. Additionally, a Variational Autoencoder model will be implemented to evaluate the automatic extraction of relevant features (latent space) as an alternative to manual feature engineering. Despite the dataset being labeled, the problem is treated as unsupervised, reflecting the real-world scenario where labels are often unavailable. Labels are used only for validation purposes after model training.

### Data Availability
The dataset used in this study is publicly available at Bosch Research CNC Machining Dataset (https://github.com/boschresearch/CNC_Machining).

#### References
Tnani, M., Feil, P., & Diepold, K. (2022). Efficient Feature Learning Approach for Raw Industrial Vibration Data Using Two-Stage Learning Framework. Journal of Manufacturing Processes.
