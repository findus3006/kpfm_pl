"""Qt-free analysis models behind the GUI tabs.

Each module wraps the public ``kpfm_pl`` API for one workflow: it holds the
parameters, performs the backend calls in the right order and returns plain
results. Keeping Qt out of here makes the models testable with plain pytest.
"""
