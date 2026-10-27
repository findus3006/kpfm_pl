@echo off
rem Start the kpfm_pl GUI from this folder (double-click or run from cmd).
rem Uses the code in .\src, so kpfm_pl itself need not be installed; its
rem dependencies are needed: pip install -e .[gui]
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

python -c "import kpfm_pl.gui.app" || (
    echo.
    echo The kpfm_pl GUI could not be started; see the error above.
    echo Install Python and the GUI dependencies in this folder with:
    echo     pip install -e .[gui]
    echo.
    pause
    exit /b 1
)

rem pythonw starts the window without keeping a console open.
start "" pythonw -m kpfm_pl.gui
