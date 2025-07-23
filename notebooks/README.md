# 📓 Notebooks Directory

This directory contains practical implementations of Neural Network-based GARCH calibration methods. Each notebook demonstrates a complete workflow from data processing to real-world financial applications.

---

## 🧠 Available Notebooks

### 1. GARCH Calibration with Neural Networks
[`garch_calibration.ipynb`](./garch_calibration.ipynb)  
*End-to-end workflow for calibrating GARCH parameters using neural networks*

Key features:
- Data preprocessing for financial time series
- Neural network architecture design (LSTM/GRU)
- Model training and validation
- Real-time calibration on streaming data
- Performance benchmarking vs traditional methods

```python
# Core calibration workflow
import tensorflow as tf
from arch import arch_model

# Neural network calibration
nn_model = tf.keras.Sequential([...])
garch_params = nn_model.predict(streaming_data)

# Feed to stochastic model
heston_model.calibrate(initial_params=garch_params[['VL','persistence']])

# Portfolio optimization
optimizer.run(volatility_forecast=garch_params['conditional_volatility'])
