"""READ-Grid rebuild package (P359).

Modules
-------
protocol      frozen protocol loading
mapping       91 feeder loads -> 50 logical groups -> 5 edge domains
feeder        OpenDSS wrapper (compile, patch, set loads, solve, measure)
workload      24 h workload episodes from public IEEE LV profiles
reward        locally-observable learning reward
value_model   corrected action-dependent linear model + concatenated ablation
controllers   C0..C5
arbiter       authority / command-fence state machine
comm          event-driven communication model (SimPy clock)
faults        F0..F10 fault schedules
sim           multirate closed-loop simulator
metrics       metric extraction
"""

__all__ = [
    "protocol",
    "mapping",
    "feeder",
    "workload",
    "reward",
    "value_model",
    "controllers",
    "arbiter",
    "comm",
    "faults",
    "sim",
    "metrics",
]
