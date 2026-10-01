# BEV energy dynamics dataset

This repository contains scripts and datasets associated with the paper:

 **Unveiling Energy Dynamics of Battery Electric Vehicles Using High-Resolution Data**  
Authors: M. Yasko, A. Moussa, F. Tian, H. Kazmi, J. Driesen, and W. Martinez  
Institution: KU Leuven / EnergyVille, Thor Park 8310, Genk, Belgium  
Last updated: 2025-06-12  
Contact: mohamed.yasko@kuleuven.be

---

## 📁 Repository Structure

### 1. Main Data Folders

These folders contain pre-processed `.csv` files categorized by session type:

- `slow charging sessions/` — Slow AC charging session files
- `fast charging sessions/` — Fast DC charging session files
- `driving sessions/` — Vehicle driving session files
- `parking sessions/` — Parking session files (idle with auxiliary activity)

### 2. Additional Resources

- `totalenergy_2years/` — Energy consumption and odometer data over a two-year period, separated by battery modules
- `000001.mf4` — Example of a raw data file recorded by the measurement device (ASAM MDF format)
- `000001.csv` — Example of a decoded and processed CSV file
- `CAN.dbc` — DBC file used for decoding CAN messages in the raw `.mf4` file
- `data_illustration/` — Python scripts for generating figures and illustrative plots used in the publication
- `data_processing/` — Python scripts for processing and structuring raw or pre-processed EV data
- `pylibs/` — A reference list of all Python libraries used in the project
-List of variable containing all the possible variables.

---

## How to Use

1. Place raw `.mf4` files in a directory of your choice.
2. Use `data_processing/` scripts to decode and format the data using the provided `CAN.dbc`.
3. Analyze the structured `.csv` files by session type using the scripts in `data_illustration/`.

---

## License

This repository and its content are shared under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

