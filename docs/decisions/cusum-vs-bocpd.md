# Decision: CUSUM (hand-implemented) over BOCPD

Chose two-sided CUSUM (Page 1954) for online detection. Rejected BOCPD (Adams & MacKay 2007).

BOCPD is Bayesian, heavier, and harder to explain. CUSUM is minimax-optimal for detection delay (Moustakides 1986), O(1) per observation, and ~40 lines of pure Python. Hand-implementing it shows depth — same reason PSI was hand-implemented in V1.

ruptures provides no online detector; BOCPD would be a new dependency with harder calibration. CUSUM with empirical h calibration is the honest, explainable route.
