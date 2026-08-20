# Selective-Algorithm-for-Thermal-Target-Tracking
A C++ engine for real-time thermal target tracking with zero dependencies and deterministic results. Uses a 12x12 custom morphological contrast kernel ($C_k$) and predictive ROI gating ($H$) to achieve &lt;10ms frame latency (>100 FPS) and &lt;20MB RAM on standard CPUs while removing cloud clutter false alarms.
