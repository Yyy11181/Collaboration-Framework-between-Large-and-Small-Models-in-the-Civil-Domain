# Collaboration Framework between Large and Small Models in the Civil Domain

## Project Structure

### 📁 Directories
| Directory | Description |
|-----------|------------|
| `civil_data/` | Contains all datasets including:<br>- Training data<br>- Validation data<br>- Test data <br>- dataprocessing file<br>|
| `civil_output/` |  <br>- SMs prediction outputs during the testing phase <br>- The database used for LLMs's secondary prediction<br>|

### 📋 Key Scripts
| Script | Purpose |
|--------|---------|
| `api_my.py` | The used function in the LLMs API interface implementation |
| `cm_analyse.py` | The script for the confusing charge dictionary  |
| `confidence.py` | Threshold determination based on maximum predicted probability<br>*(Adjust threshold for optimal performance)* |
| `main_multi_new_civil.py` | Main prediction script for SMs |
| `prompt_civil_based.py` |LLMs prediction handler |
| `prompt_civil.py` | LLMs secondary prediction handler (for SMs collaboration) |
| `train_civil.py` | SMs training script<br>*(Automatically saves best-performing model)* |
| `util_civil.py` | Utilities for:<br>- Data preprocessing<br>- Performance metric computation |

## Key Features
✔ Hybrid architecture combining LLMs and SMs  
✔ Candidate threshold determination  
✔ Modular design for easy extension  
✔ Comprehensive performance tracking

> **Note**: All paths are relative to the project root. Requires Python 3.8+.