# Edit on macOS, Train on a Remote Windows GPU Laptop

## How the pieces fit

| Where | Role |
|---|---|
| MacBook | The only place where code is edited and committed. Unit tests and tiny pilots run here. |
| GitHub | Carries code from the Mac to Windows (`git push` → `git pull`). |
| Windows laptop (RTX 5070 Ti) | A clone that only pulls. It runs training and evaluation, and keeps `runs/` and `logs/`. |
| VS Code | One local window for editing, and one **Remote-SSH** window on the Windows laptop for terminals, logs and `nvidia-smi`. |

Every run appends the host, GPU, PyTorch version and git commit (plus a dirty flag) to `runs/<name>/seed<k>/run_info.jsonl`, so each result can be traced to the exact code that produced it.

**Why runs need a launcher.** Windows OpenSSH ends every process of a session when the connection drops, and Windows has no `nohup` or `tmux`. A training run started directly in an SSH terminal therefore dies when the Mac sleeps or the network blips. `scripts/launch_windows.ps1` starts runs through WMI instead, so they survive disconnects.

## One-time setup

### On the Mac

1. Install the VS Code extension **Remote - SSH** (`ms-vscode-remote.remote-ssh`).
2. Add the laptop to `~/.ssh/config`:

   ```
   Host avalon-gpu
       HostName <IP address or hostname of the Windows laptop>
       User <Windows user name>
   ```

3. In VS Code, run **Remote-SSH: Connect to Host…** → `avalon-gpu`. Choose **Windows** when VS Code asks for the platform.
4. The terminal in that window is PowerShell on the laptop. Every remaining setup step below runs there.

### On the Windows laptop

1. **GPU driver.** Run `nvidia-smi`; the reported CUDA version must be 12.8 or newer, which RTX 50-series GPUs require. Update the NVIDIA driver if it is older.

2. **Git and conda.** Install Git for Windows and Anaconda (or Miniconda), e.g. `winget install --id Git.Git -e`. Run `conda init powershell` once, so that `conda activate` works in PowerShell terminals.

3. **Access to the private repository.** Use a read-only deploy key, so the laptop can pull but never push.

   ```powershell
   ssh-keygen -t ed25519 -C "avalon-gpu deploy key" -f $HOME\.ssh\id_ed25519_github
   Get-Content $HOME\.ssh\id_ed25519_github.pub
   ```

   Add the printed key on GitHub under the repository's **Settings → Deploy keys** (leave write access off). Then tell SSH to use it for GitHub by appending this to `$HOME\.ssh\config`:

   ```
   Host github.com
       HostName github.com
       User git
       IdentityFile ~/.ssh/id_ed25519_github
       IdentitiesOnly yes
   ```

   Check with `ssh -T git@github.com`; the reply should greet the repository. In scripted, non-interactive SSH sessions, Windows `ssh` waits on standard input forever, so run it as `cmd /c "ssh -T git@github.com < NUL"`. An interactive VS Code terminal is not affected. Then clone:

   ```powershell
   cd $HOME
   git clone git@github.com:zhiqiH/Avalon-Signals.git
   cd Avalon-Signals
   ```

4. **Python environment.** The conda environment is named after the project, `avalon-signals`, on every machine. conda-forge avoids the Terms-of-Service prompt that Anaconda's default channels show in non-interactive sessions.

   ```powershell
   conda create -y -n avalon-signals --override-channels -c conda-forge python=3.12
   conda activate avalon-signals
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install -e ".[dev]"
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   pytest
   python scripts\benchmark.py --games 2048 --device cuda
   ```

   In the Remote-SSH window, install the VS Code **Python** extension. Then select the interpreter of the `avalon-signals` conda environment, e.g. `C:\Users\<user>\anaconda3\envs\avalon-signals\python.exe`. The launcher finds this environment by name; it does not need to be activated.

5. **Power.** While training, keep the laptop plugged in and stop it from sleeping:

   ```powershell
   powercfg /change standby-timeout-ac 0
   ```

   Later, `powercfg /change standby-timeout-ac 30` restores a 30-minute timeout.

## Daily loop

1. **Mac:** edit the code, run `pytest`, commit, and `git push`.

2. **Windows** (terminal of the Remote-SSH window, after `conda activate avalon-signals`):
   1. Get the new code with `git pull`.
   2. On the first use, check that the launcher works with a tiny run:

      ```powershell
      powershell -ExecutionPolicy Bypass -File scripts\launch_windows.ps1 -Config configs\pilot_silent.yaml -Seeds 0 -ExtraArgs "--total-games 512"
      ```

   3. Start the real runs, one process per seed:

      ```powershell
      powershell -ExecutionPolicy Bypass -File scripts\launch_windows.ps1 -Seeds 0,1,2
      ```

      The launcher prints each run's PID and log file. Once started, the runs no longer depend on the SSH connection. Running `python scripts\train.py ...` directly in the terminal is fine for quick checks, but it stops when the connection drops.

3. **Monitor** a run in any of these ways:
   - follow its log: `Get-Content logs\<log file> -Wait -Tail 5`;
   - open `runs\<name>\seed<k>\metrics.jsonl` in VS Code;
   - watch the GPU with `nvidia-smi`;
   - list running jobs: `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine`;
   - stop a run with `taskkill /PID <pid> /T /F`.

4. **Evaluate** on the laptop:

   ```powershell
   python scripts\evaluate.py --checkpoint runs\baseline_silent\seed0\latest.pt --games 1000
   ```

5. **Bring results to the Mac** from a Mac terminal:

   ```bash
   scp -r avalon-gpu:Avalon-Signals/runs/baseline_silent ./runs/
   ```

   If the relative path is not found, use the absolute form `avalon-gpu:C:/Users/<user>/Avalon-Signals/runs/baseline_silent`. Commit only small curated outputs, such as evaluation JSON and figures, under `results/`; `runs/` and `logs/` stay out of git.

## Rules of thumb

- **Edit in one place only.** If you change code on Windows in an emergency, commit and push it there with your own credentials, then pull on the Mac before editing again.
- **Line endings.** `.gitattributes` keeps LF line endings on both machines, so a pull never rewrites every file.
- **Where speed comes from.** Rollouts are CPU-bound and the GPU only speeds up the PPO update, so throughput comes from parallel seeds. Use `--threads 1` or `--threads 2` per run (the launcher's `-Threads`), and run no more seeds than the laptop has physical cores.
