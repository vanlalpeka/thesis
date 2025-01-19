This is the code for my master's thesis: <a href="Master Thesis with affidavit.pdf">Anomaly Detection Using an Ensemble with Simple Sub-models, 2024</a>.
The algorithm explores the effectiveness of an ensemble of simple sub-models like linear regression in detecting anomalies.

How to pull from the GitHub Container Registry:
```
docker pull ghcr.io/vanlalpeka/msc_thesis:latest
```


How to test:
- First, execute job_<dataset>.py to run the algorithm. The parameters for each run are stored in the params folder. 
- Second, execute competitors.py to run KNN, Isolation Forest, and CBLOF from the pyod package.
- Finally, compare the AUROCs from the two steps above to gauge the performance.

Here's a high-level history of the code versions.
![image](https://github.com/user-attachments/assets/efe62fdd-b569-4f3c-bd33-9620e6f1c0b7)
