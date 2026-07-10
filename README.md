# Autonomous Robots (Webots)

Short setup and run guide.

## Requirements

- Windows + PowerShell
- Python installed and available in PATH
- Webots installed

## Setup (run once)

From the project root, run:

```powershell
./setup.ps1
```

What this does:

- creates `.venv` (if missing)
- upgrades pip
- installs packages from `requirements.txt`
- registers `sys-code` as an import path via a `.pth` file

## Configure each controller to use `.venv`

In each controller folder for each Maze (Maze1 to Maze5), set `runtime.ini` like this:

```ini
[python]
COMMAND = "C:/Users/<YOUR_USER>/SebastianWeidnerModularbeit/.venv/Scripts/python.exe"
```

Example file:

- `sys-code/Maze4/controllers/Controller_v1/runtime.ini`

Do this for every controller you run.

## Run in Webots

1. Open a world (for example in `sys-code/Maze4/worlds/`).
2. Make sure the robot controller points to the controller folder with the updated `runtime.ini`.
3. Start the simulation.

## Notes

- If dependencies change, rerun `./setup.ps1`.
- If `.venv` gets broken, delete it and run `./setup.ps1` again.
