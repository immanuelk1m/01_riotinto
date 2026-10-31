# colab CLI documentation (captured 2026-10-31T10:35:26+09:00)

## colab -h
                                                                                
 Usage: colab [OPTIONS] COMMAND [ARGS]...                                       
                                                                                
 Colab CLI                                                                      
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --client-oauth-config  -c      <str>         Path to client OAuth config     │
│                                              JSON file                       │
│                                              [default:                       │
│                                              /Users/sonet/.colab-cli-oauth-… │
│ --config                       <str>         Path to session state file      │
│                                              (~/.config/colab-cli/sessions.… │
│ --logtostderr                                Log all output to stderr        │
│ --auth                         <oauth2|adc>  Authentication strategy to use: │
│                                              'oauth2' (public                │
│                                              InstalledAppFlow), or 'adc'     │
│                                              (Application Default            │
│                                              Credentials).                   │
│                                              [default: oauth2]               │
│ --install-completion                         Install completion for the      │
│                                              current shell.                  │
│ --show-completion                            Show completion for the current │
│                                              shell, to copy it or customize  │
│                                              the installation.               │
│ --help                 -h                    Show this message and exit.     │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ console         Connect to raw TTY console                                   │
│ download        Download a file from a session                               │
│ drivemount      Mount Google Drive at path                                   │
│ edit            Edit a file on a running Colab session                       │
│ exec            Execute code in a session                                    │
│ help            Show help for a command.                                     │
│ install         Install python packages on the VM                            │
│ log             Manage and view session history logs                         │
│ ls              List files in a session                                      │
│ new             Create a new session                                         │
│ pay             Open the Colab signup page to manage compute units           │
│ readme          Print the bundled README.md file                             │
│ repl            Start an interactive REPL                                    │
│ restart-kernel  Restart a session's kernel                                   │
│ rm              Remove a remote file                                         │
│ run             Run a Python script on a fresh Colab VM, then release the VM │
│ sessions        List all active sessions                                     │
│ skill           Print the bundled SKILL.md file                              │
│ ssh             Connect to a Colab runtime via SSH.                          │
│ status          Show session status                                          │
│ stop            Stop a session                                               │
│ update          Check for latest version and print if an update is available │
│ upload          Upload a file to a session                                   │
│ url             Print a browser URL that connects to an existing session.    │
│ usage           Show account compute-unit usage rate and balance             │
│ version         Show the version of the Colab CLI                            │
╰──────────────────────────────────────────────────────────────────────────────╯



## colab help new
                                                                                
 Usage: colab new [OPTIONS]                                                     
                                                                                
 Create a new session                                                           
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session   -s      <str>  Session name                                      │
│ --tpu               <str>  TPU accelerator variant. Supported: v5e1, v6e1.   │
│ --gpu               <str>  GPU accelerator variant. Supported: T4, L4, G4,   │
│                            H100, A100.                                       │
│                            If omitted (along with --tpu), a CPU runtime is   │
│                            created.                                          │
│                            Availability varies by Colab subscription tier.   │
│ --high-mem                 Request a high-RAM machine shape (CPU, T4, A100,  │
│                            etc.). Requires Colab Pro or Pro+ entitlement.    │
│                            Ignored for accelerators that only offer a single │
│                            shape (L4, v5e1, v6e1).                           │
│ --help      -h             Show this message and exit.                       │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help exec
                                                                                
 Usage: colab exec [OPTIONS]                                                    
                                                                                
 Execute code in a session                                                      
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session       -s      <str>    Session name                                │
│ --file          -f      <str>    File to execute                             │
│ --output-image          <str>    Path to save plot                           │
│ --timeout               <float>  Timeout in seconds for code execution       │
│                                  [default: 30.0]                             │
│ --env                   <str>    Set an environment variable in the remote   │
│                                  kernel as KEY=VALUE. Repeat for multiple    │
│                                  variables.                                  │
│ --help          -h               Show this message and exit.                 │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help console
                                                                                
 Usage: colab console [OPTIONS]                                                 
                                                                                
 Connect to raw TTY console                                                     
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session  -s      <str>  Session name                                       │
│ --help     -h             Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help upload
                                                                                
 Usage: colab upload [OPTIONS] {local_path} {remote_path}                       
                                                                                
 Upload a file to a session                                                     
                                                                                
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    local_path       <str>  Local file to upload [required]                 │
│ *    remote_path      <str>  Remote path to upload to [required]             │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session  -s      <str>  Session name                                       │
│ --help     -h             Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help download
                                                                                
 Usage: colab download [OPTIONS] {remote_path} {local_path}                     
                                                                                
 Download a file from a session                                                 
                                                                                
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    remote_path      <str>  Remote path to download from [required]         │
│ *    local_path       <str>  Local path to save the file [required]          │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session  -s      <str>  Session name                                       │
│ --help     -h             Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help status
                                                                                
 Usage: colab status [OPTIONS]                                                  
                                                                                
 Show session status                                                            
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session  -s      <str>  Session name                                       │
│ --help     -h             Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help stop
                                                                                
 Usage: colab stop [OPTIONS]                                                    
                                                                                
 Stop a session                                                                 
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session  -s      <str>  Session name                                       │
│ --help     -h             Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help sessions
                                                                                
 Usage: colab sessions [OPTIONS]                                                
                                                                                
 List all active sessions                                                       
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help run
                                                                                
 Usage: colab run [OPTIONS] {script} [script_args]...                           
                                                                                
 Run a Python script on a fresh Colab VM, then release the VM                   
                                                                                
 Designed to be used as a shebang interpreter, e.g.                             
                                                                                
     #!/usr/bin/env -S colab run --gpu T4                                       
                                                                                
 so a single executable .py file can rent a GPU, run, and clean up after        
 itself.                                                                        
                                                                                
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    script           <str>  Path to a local Python file to execute on a     │
│                              fresh Colab VM.                                 │
│                              [required]                                      │
│      script_args      <str>  Arguments forwarded to the script as            │
│                              sys.argv[1:]. Anything after the script path is │
│                              passed through verbatim.                        │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session   -s      <str>    Name for the ephemeral session (auto-generated  │
│                              if omitted). Useful with --keep so you can      │
│                              attach later via `colab exec -s <name>`.        │
│ --tpu               <str>    TPU accelerator variant. Supported: v5e1, v6e1. │
│ --gpu               <str>    GPU accelerator variant. Supported: T4, L4, G4, │
│                              H100, A100. If omitted (along with --tpu), a    │
│                              CPU runtime is created.                         │
│ --high-mem                   Request a high-RAM machine shape. Requires      │
│                              Colab Pro or Pro+ entitlement. Ignored for L4   │
│                              and TPU accelerators.                           │
│ --keep                       Do not stop the session after the script        │
│                              finishes. The session remains in `colab         │
│                              sessions` until you run `colab stop`.           │
│ --timeout           <float>  Timeout in seconds for code execution           │
│                              [default: 30.0]                                 │
│ --env               <str>    Set an environment variable in the remote       │
│                              kernel as KEY=VALUE. Repeat for multiple        │
│                              variables.                                      │
│ --help      -h               Show this message and exit.                     │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help ssh
                                                                                
 Usage: colab ssh [OPTIONS]                                                     
                                                                                
 Connect to a Colab runtime via SSH.                                            
                                                                                
 Bare ``colab ssh`` uses your only active session, or auto-creates one (like    
 ``colab new``) if you have none, and opens a shell in ``/content``. With       
 --proxy-mode it is a ProxyCommand-compatible WebSocket-stdio bridge; every     
 flag above still applies (``-s NAME`` creates the session if missing,          
 ``--gpu/--tpu`` set its accelerator, ``--rm`` stops it on disconnect).         
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session     -s      <str>  Session name                                    │
│ --proxy-mode                 Act as an OpenSSH ProxyCommand-compatible       │
│                              WebSocket-stdio bridge (reads stdin, writes     │
│                              stdout). Use in ~/.ssh/config as `ProxyCommand  │
│                              colab ssh --proxy-mode -s SESS`. All flags      │
│                              below also apply here.                          │
│ --identity    -i      <str>  SSH private key whose public key is sent in the │
│                              X-Colab-Ssh-Pubkey header (default: first of    │
│                              ~/.ssh/id_ed25519, id_ecdsa).                   │
│ --gpu                 <str>  GPU accelerator for a runtime created by this   │
│                              command (T4, L4, G4, H100, A100). Used when the │
│                              session is auto-created.                        │
│ --tpu                 <str>  TPU accelerator for a runtime created by this   │
│                              command (v5e1, v6e1). Used when the session is  │
│                              auto-created.                                   │
│ --high-mem                   Request a high-RAM machine shape when this      │
│                              command auto-creates a runtime.                 │
│ --rm                         Stop the runtime when the session ends. In      │
│                              interactive mode this applies only to a runtime │
│                              `colab ssh` auto-created; in --proxy-mode it    │
│                              stops the bridged session on disconnect         │
│                              (ephemeral ~/.ssh/config host).                 │
│ --help        -h             Show this message and exit.                     │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help install
                                                                                
 Usage: colab install [OPTIONS] [packages]...                                   
                                                                                
 Install python packages on the VM                                              
                                                                                
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│   packages      <str>  Packages to install                                   │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session      -s      <str>  Session name                                   │
│ --requirement  -r      <str>  Requirements file                              │
│ --help         -h             Show this message and exit.                    │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help log
                                                                                
 Usage: colab log [OPTIONS]                                                     
                                                                                
 Manage and view session history logs                                           
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session  -s      <str>  Session name (if omitted, lists all sessions with  │
│                           logs)                                              │
│ --lines    -n      <int>  Number of lines to show/export (default: all)      │
│ --type     -t      <str>  Filter by event type (e.g., execution,             │
│                           file_operation)                                    │
│ --output   -o      <str>  Output file path (suffix determines format:        │
│                           .ipynb, .md, .txt, .jsonl)                         │
│ --help     -h             Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help ls
                                                                                
 Usage: colab ls [OPTIONS] [path]                                               
                                                                                
 List files in a session                                                        
                                                                                
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│   path      <str>  Remote path to list [default: content]                    │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session  -s      <str>  Session name                                       │
│ --help     -h             Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help rm
                                                                                
 Usage: colab rm [OPTIONS] {path}                                               
                                                                                
 Remove a remote file                                                           
                                                                                
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    path      <str>  Remote path to remove [required]                       │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session  -s      <str>  Session name                                       │
│ --help     -h             Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help url
                                                                                
 Usage: colab url [OPTIONS]                                                     
                                                                                
 Print a browser URL that connects to an existing session.                      
                                                                                
 Format: ``https://<host>/notebooks/empty.ipynb?dbu=<urlencoded                 
 path>#datalabBackendUrl=<host>/tun/m/<endpoint>``,                             
 where the path is ``/tun/m/<endpoint>``. When opened, the Colab frontend       
 skips ``/tun/m/assign`` and attaches the kernel to our existing VM.            
                                                                                
 Two backend-URL signals are embedded:                                          
                                                                                
 - ``?dbu=<urlencoded path>`` — the ``datalab_backend_url`` development         
   query flag. The frontend resolves the value against                          
   ``window.location.origin``.                                                  
                                                                                
 - ``#datalabBackendUrl=<full URL>`` — the hash-fragment form. Some             
   frontend code paths consult this first and ignore ``dbu``, so we             
   emit both for robustness. The fragment value is a FULL URL (with             
   scheme + host) and is intentionally NOT URL-encoded — browsers do            
   not decode the fragment before passing ``location.hash`` to page             
   JS, and Colab's hash parser expects the raw string.                          
                                                                                
 The fragment's host always matches ``--host`` (the page origin), so            
 Colab's same-origin enforcement on the embedded backend URL doesn't            
 block the connection, and sandbox/dev users get a sandbox fragment             
 automatically.                                                                 
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session  -s      <str>  Session name                                       │
│ --host             <str>  Colab frontend host (origin) to use for the URL.   │
│                           The Colab frontend resolves `dbu` against          │
│                           `window.location.origin`, so this only changes the │
│                           page origin, not the embedded backend path.        │
│                           [default: https://colab.research.google.com]       │
│ --open                    After printing the URL, also open it in the system │
│                           browser. Off by default so the command remains     │
│                           pipeable (e.g. `colab url -s s1 | xclip`).         │
│ --help     -h             Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help usage
                                                                                
 Usage: colab usage [OPTIONS]                                                   
                                                                                
 Show account compute-unit usage rate and balance                               
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help readme
                                                                                
 Usage: colab readme [OPTIONS]                                                  
                                                                                
 Print the bundled README.md file                                               
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help skill
                                                                                
 Usage: colab skill [OPTIONS]                                                   
                                                                                
 Print the bundled SKILL.md file                                                
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help restart-kernel
                                                                                
 Usage: colab restart-kernel [OPTIONS]                                          
                                                                                
 Restart a session's kernel                                                     
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session  -s      <str>  Session name                                       │
│ --help     -h             Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help edit
                                                                                
 Usage: colab edit [OPTIONS] {remote_path}                                      
                                                                                
 Edit a file on a running Colab session                                         
                                                                                
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    remote_path      <str>  Remote path to edit [required]                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session  -s      <str>  Session name                                       │
│ --help     -h             Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help drivemount
                                                                                
 Usage: colab drivemount [OPTIONS] [path]                                       
                                                                                
 Mount Google Drive at path                                                     
                                                                                
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│   path      <str>  Mount path [default: /content/drive]                      │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session  -s      <str>  Session name                                       │
│ --help     -h             Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help repl
                                                                                
 Usage: colab repl [OPTIONS]                                                    
                                                                                
 Start an interactive REPL                                                      
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session       -s      <str>  Session name                                  │
│ --output-image          <str>  Path to save plot                             │
│ --help          -h             Show this message and exit.                   │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help pay
                                                                                
 Usage: colab pay [OPTIONS]                                                     
                                                                                
 Open the Colab signup page to manage compute units                             
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help update
                                                                                
 Usage: colab update [OPTIONS]                                                  
                                                                                
 Check for latest version and print if an update is available                   
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --install            After checking, run 'pip install -U google-colab-cli'   │
│                      to upgrade the CLI in place. No-op if already up to     │
│                      date. Linux only.                                       │
│ --help     -h        Show this message and exit.                             │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab help version
                                                                                
 Usage: colab version [OPTIONS]                                                 
                                                                                
 Show the version of the Colab CLI                                              
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯


## colab readme
# Colab CLI

A command-line interface for Google Colab. Provision high-performance CPU, GPU, and TPU runtimes, execute local code, manage remote files, and orchestrate automated cloud pipelines — directly from your terminal.

Designed to support seamless developer productivity, headless automation, and AI agent integrations.

[Demo](https://github.com/user-attachments/assets/656226a9-af13-4fdb-8eda-d7de747336a2)

> [!NOTE]
> **Platform support:** the Colab CLI currently supports **Linux and macOS** only. Windows is not supported at this time.

> [!TIP]
> Looking for in-notebook, interactive agent-assisted coding instead of a terminal workflow? See the [Colab MCP Server](https://github.com/googlecolab/colab-mcp).

---

## Key Features

* **Instant VM Provisioning:** Spin up CPU, GPU (T4, L4, G4, H100, A100), or TPU (v5e1, v6e1) runtimes in seconds.
* **Robust Code Execution:** Run local Python scripts, Jupyter Notebooks (`.ipynb`), or piped `stdin` code; launch interactive REPLs or raw TTY console shells.
* **Ephemeral Job Runner (`colab run`):** Provision a fresh VM, execute a local script with forwarded arguments, retrieve output files, and automatically tear down the runtime in a single command.
* **Activity-Based Liveness:** Sessions stay alive as long as the kernel is active, with the Colab backend automatically maintaining liveness.
* **Seamless Workspace Automation:** Mount Google Drive, authenticate Google Cloud Platform (GCP) credentials, and install dependencies with high-performance `uv` package management.
* **State & Log Archival:** Inspect local session states or export interactive history logs to standard Jupyter Notebooks, Markdown, or structured JSONL.

---

## Installation

Install the package using `uv` (recommended) or standard `pip`:

```bash
# Using uv (recommended)
uv tool install google-colab-cli

# Using pip
pip install google-colab-cli
```

---

## Quick Start

Run a CPU-based VM runtime, execute some code, and clean up:

```bash
# 1. Provision a new session
colab new

# 2. Execute code from stdin
echo 'print("Hello from Google Colab!")' | colab exec

# 3. Stop and release the VM resource
colab stop
```

> [!NOTE]
> When only one session is active, you can omit the `-s, --session` option;
> the CLI automatically knows it.


---

## Command Index

Run `colab <command> --help` to view specific options, defaults, and detailed help.

### Session Management
| Command | Description |
| --- | --- |
| `colab new [-s NAME] [--gpu GPU] [--tpu TPU] [--high-mem]` | Allocate a new CPU, GPU, or TPU VM runtime (optionally high-RAM) |
| `colab sessions` | List all active sessions currently active on the backend |
| `colab status [-s NAME]` | Display hardware, machine shape, status, and local metadata for active sessions |
| `colab restart-kernel [-s NAME]` | Restart the active session's Jupyter kernel |
| `colab stop [-s NAME]` | Terminate a session VM and release its resources |
| `colab url [-s NAME] [--open]` | Print or open a browser URL connecting to the active session |

### Execution
| Command | Description |
| --- | --- |
| `colab run [--gpu GPU] [--tpu TPU] [--high-mem] [--keep] SCRIPT [ARGS...]` | Run a local script on a fresh VM, forwarding arguments, then release it |
| `colab exec [-s NAME] [-f FILE] [--output-image PATH]` | Execute Python code from stdin, a local `.py` file, or a `.ipynb` notebook |
| `colab repl [-s NAME] [--output-image PATH]` | Start an interactive Python REPL on the VM (exits cleanly on piped EOF) |
| `colab console [-s NAME]` | Connect to a raw interactive TTY shell (tmux) on the remote VM |
| `colab ssh [-s NAME] [--proxy-mode] [-i KEY] [--gpu GPU] [--tpu TPU] [--high-mem]` | Open an SSH shell to the runtime over WebSocket, or act as an OpenSSH `ProxyCommand` bridge for IDE remote-dev |

### File Operations
| Command | Description |
| --- | --- |
| `colab ls [-s NAME] [PATH]` | List remote files on the VM |
| `colab upload [-s NAME] LOCAL REMOTE` | Upload a local file to the VM filesystem |
| `colab download [-s NAME] REMOTE LOCAL` | Download a remote file from the VM filesystem |
| `colab rm [-s NAME] PATH` | Delete a remote file on the VM filesystem |
| `colab edit [-s NAME] PATH` | Edit a remote file in-place using your local `$EDITOR` |

### Automation & Utilities
| Command | Description |
| --- | --- |
| `colab auth [-s NAME]` | Authenticate the VM for GCP services (BigQuery, GCS, etc.) |
| `colab drivemount [-s NAME] [PATH]` | Mount Google Drive on the VM (default: `/content/drive`) |
| `colab install [-s NAME] [-r FILE \| PKG...]` | Install packages on the VM using `uv` (falls back to `pip`) |
| `colab log [-s NAME] [-n N] [-o FILE]` | View or export session history (`.ipynb`, `.md`, `.txt`, `.jsonl`) |
| `colab usage` | Show account compute-unit usage rate and balance |
| `colab pay` | Open the Colab subscription page to manage compute units |
| `colab version` | Print the installed version of the CLI |
| `colab update [--install]` | Check for a newer release (and optionally upgrade the CLI in place) |

### Global Options
* `--auth {oauth2,adc}` — Authentication strategy for the Colab API (default: `adc`).
* `-c, --client-oauth-config PATH` — Path to public OAuth client credentials configuration (default: `~/.colab-cli-oauth-config.json`).
* `--config PATH` — Path to local session metadata storage (default: `~/.config/colab-cli/sessions.json`).
* `--logtostderr` — Direct debug logging output to stderr.

---

## Practical Examples

### Accelerator Training with Checkpoint Retrieval

Provision an A100 GPU, install requirements, run a local training script, retrieve the resulting model weights, and terminate the VM:

```bash
colab new -s trainer --gpu A100
colab install -s trainer torch transformers
colab exec -s trainer -f train.py
colab download -s trainer checkpoints/model.bin ./model.bin
colab stop -s trainer
```

### Workspace Notebook Execution with Drive Integration

Mount Google Drive, run a local notebook against the VM kernel (outputs are written back into `report_output.ipynb`), export a Markdown log of the execution, and clean up:

```bash
colab new -s analysis
colab drivemount -s analysis
colab exec -s analysis -f report.ipynb
colab log -s analysis -o execution_log.md
colab stop -s analysis
```

---

## Usage Notes

* **Machine shape:** Use `--high-mem` with `colab new`, `colab run`, or `colab ssh` (when auto-creating a runtime) to request a high-RAM machine shape. Requires Colab Pro or Pro+ entitlement for supported accelerators (CPU, T4, A100, etc.). L4 and TPU runtimes ignore this flag because they only offer one shape. Machine shape is shown in `colab sessions` and `colab status`.
* **TTY Requirements:** The interactive commands `repl` and `console` require a local TTY. When running inside automated scripts or pipelines, make sure to pipe stdin (e.g., `echo "print(1)" | colab repl`) to trigger non-interactive execution modes.
* **Transparent Code Execution:** When calling `colab exec -f file.py`, the CLI reads the file locally and transmits its content to the remote kernel. You do not need to manually upload files before execution.
* **Storage & State Paths:** Session tokens and metadata are stored at `~/.config/colab-cli/sessions.json`. Global CLI settings are located at `~/.config/colab-cli/settings.json`. These can be customized or isolated via the global `--config` flag.

### Ephemeral Accelerator Jobs

Use `colab run` to run a local script on dedicated hardware without manual session lifecycle management. The CLI handles provisioning, script execution, and immediate VM teardown automatically:

```bash
# Run train.py on a T4 GPU and release the VM on completion
colab run --gpu T4 train.py
```

### Shebang Execution Support

To execute a local file directly on a remote accelerator, place the `colab run` interpreter in the shebang line:

```python
#!/usr/bin/env -S colab run --gpu L4 --keep
import torch

print("L4 GPU Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0))
```

Make the script executable (`chmod +x script.py`) and run it: `./script.py`. The `--keep` option tells the CLI to preserve the session VM on completion so you can re-execute or inspect logs.

---

## Deep Dive Documentation

For comprehensive architectural overviews and deep-dives into specific CLI sub-systems, refer to the detailed documentation:

* [Session Management Architecture](docs/01_session_management.md)
* [Interactive & Non-Interactive Execution Design](docs/02_execution_and_interactive.md)
* [File Management & Jupyter Contents API](docs/03_file_management.md)
* [Authentication Providers & VM Automation](docs/04_automation_and_utility.md)
* [Ephemeral Job Runner Design](docs/05_run_command.md)

To view interactive walkthroughs of eleven real-world automated scenarios, check out the [Demo Walkthroughs](docs/demos.md).

---

## Contributing

Feedback and contributions are welcome! Please read [`CONTRIBUTING.md`](./CONTRIBUTING.md) for details.

---

## Legal & Privacy

See Colab's [Terms of Service](https://research.google.com/colaboratory/tos_v6.html) and the [Google Privacy Policy](https://policies.google.com/privacy), which apply to usage of this CLI.


## colab skill
---
name: colab-operator
description: Operate Google Colab environments via the `colab` CLI. Use when asked to create or manage GPU/TPU sessions, run Python/shell on a remote Colab VM, sync files, automate environment setup (packages, auth, Drive), or export session history.
---

# Skill: Colab Session Operator

Operate Google Colab environments via the `colab` CLI: provision GPU/TPU sessions, run Python/shell on the VM, sync files, and capture work as notebooks.

## Installation

If the user does not already have the `colab` tool installed, it can be acquired
by running `uv tool install google-colab-cli` or `pip install google-colab-cli`.

## When to activate
- Creating or managing TPU/GPU sessions.
- Running Python or shell on a remote Colab VM.
- Syncing files between local and remote.
- Automating environment setup (packages, auth, Drive).
- Exporting session history as a Jupyter notebook.

## Mental model (read this first)
- **A session == a live Jupyter kernel on a rented VM.** `colab new` allocates a billable VM; `colab stop` releases it. Sessions stay alive as long as the kernel is active, with the Colab backend automatically maintaining liveness.
- **Kernel state PERSISTS across `colab exec` / `colab repl` calls in the same session.** Each invocation reattaches to the *same* kernel (the kernel ID is cached in local state) and only closes the websocket on exit — it does **not** shut the kernel down. So imports, variables, and defined functions survive between separate `colab exec` commands. Build up state incrementally; don't re-import everything each call. (`colab stop` and `colab restart-kernel` are what actually reset it.)
- **Default working directory is `/content`.** Every `exec`/`repl`/`run` `cd`s there first; prefer absolute paths (`/content/...`) for file work. For `colab ls/rm/upload/download`, absolute `/content/...` paths work and the default `ls` path is `content` (VM root).
- **`colab` is fire-and-forget.** Each command authenticates, does one thing, and exits.

## Authentication (the #1 thing that blocks agents)
- The global flag is `--auth={adc,oauth2}` and the **default is `adc`** (Application Default Credentials). It must come *before* the subcommand: `colab --auth=adc new -s x`.
- **ADC setup** (most reliable for headless/agent use). The Colab backends need a specific scope set, so re-mint ADC with all four scopes:
  ```bash
  gcloud auth application-default login \
    --scopes=openid,\
  https://www.googleapis.com/auth/cloud-platform,\
  https://www.googleapis.com/auth/userinfo.email,\
  https://www.googleapis.com/auth/colaboratory
  ```
  Why all four: `userinfo.email` (session backend `colab.research.google.com`, else 401), `colaboratory` (Colab features/APIs), `openid`+`cloud-platform` (mandated by gcloud itself; it rejects scope lists missing `cloud-platform`).
- **oauth2 setup**: `colab --auth=oauth2 <anything>` triggers a browser consent flow on first use (token cached at `~/.config/colab-cli/token.json`). Requires a client config at `~/.colab-cli-oauth-config.json` (or `-c PATH`). The browser step means it usually needs a human; prefer ADC for agents.
- **Verify auth in one shot**: `colab sessions` (read-only, lists server assignments) or `colab whoami` (hidden debug command: prints the active email, scopes, audience, and expiry). When any call 401s or 403s, the cause is almost always missing credentials or scopes — `colab whoami` shows it instantly.
- **Do NOT confuse `colab auth` with CLI authentication.** `colab auth` injects *VM-side* GCP credentials into the running kernel (so notebook code can call BigQuery/GCS); it is orthogonal to how the CLI itself authenticates. Never suggest "run `colab auth`" to fix a CLI 401/403 — that's a scope/identity problem fixed via the `gcloud` command above.

## Workflow

### Provision
- `colab new -s <name>` (CPU). Add `--gpu A100` or `--tpu v6e1` for accelerators. **Always pass `-s <name>`** — an omitted name is auto-generated as a random 6-hex string, which makes later commands ambiguous.
- Supported `--gpu`: `T4`, `L4`, `G4`, `H100`, `A100`. Supported `--tpu`: `v5e1`, `v6e1`.
- **Gotcha**: an unrecognized `--gpu` value silently falls back to **A100** (which then usually fails the next step). A `400` on `colab new` with an accelerator means no quota/entitlement for it on this account — fall back to `--gpu T4` or omit the flag for CPU.
- Accelerator availability is tier-gated; most accounts can only get CPU. Don't assume a GPU/TPU will allocate.

### Execute
- **Preferred**: `colab exec -s <name> -f <script.py>` runs a local script on the remote VM (read locally, sent to the kernel — no manual upload needed).
- **Piped code**: `echo "print(1)" | colab exec -s <name>` or `cat script.py | colab exec -s <name>`.
- **Notebooks**: `colab exec -s <name> -f nb.ipynb` runs each code cell and writes results to `<basename>_output.ipynb` next to the input. A `# @title Foo` first line labels the cell in progress output.
- **Plots/images**: PNG/JPEG outputs are intercepted. Use `--output-image <path>` on `exec`/`repl` to save to a known location (otherwise a temp path is printed). Inline terminal-image escapes are auto-suppressed when stdout isn't a TTY, so piped/captured output stays clean.
- **Shell**: `echo "cmd" | colab console -s <name>` for batch shell. Console wraps bash in tmux, so even piped output contains terminal-control bytes — filter with `grep -a` for a specific line. `exec` is faster when you don't need a real shell.
- **Never run `colab repl`, `colab console`, `colab auth`, or `colab drivemount` interactively from an agent** — they expect a TTY and will hang. `repl`/`console` accept piped stdin and exit on EOF; `auth`/`drivemount` genuinely require a human at the terminal.

### Ephemeral one-shot jobs (`colab run`)
- `colab run [--gpu T4] [--tpu v6e1] [--keep] [-s NAME] script.py [args...]` = `new` + `exec` + `stop` in one command. It provisions a fresh VM, runs the script with `sys.argv` and `__name__ == "__main__"` set like native `python script.py args`, then tears the VM down (unless `--keep`).
- **Exit codes propagate**: an uncaught exception or `sys.exit(N)` in the script makes `colab run` exit non-zero (CPython semantics: `sys.exit()`/`sys.exit(0)` → 0, `sys.exit(N)` → N, `sys.exit("msg")` → 1).
- **Stream separation**: `colab run` writes its own `[colab] ...` chatter to **stderr** and the script's output to **stdout** — so `colab run job.py > out.txt` captures only the script's stdout. (`colab exec` streams the script's stdout/stderr live to your stdout/stderr.)
- Works as a shebang: `#!/usr/bin/env -S colab run --gpu T4` makes a `chmod +x`'d `.py` a self-contained "rent a GPU, run, clean up" script. After editing CLI behavior, reinstall before testing shebangs — they resolve `colab` via `$PATH`, not the editable install.
- A nonexistent script path exits non-zero **before** allocating a VM (no wasted compute).

### Automate
- `colab auth -s <name>` — VM-side GCP creds, needed before in-VM GCS/BigQuery calls (interactive; not agent-runnable).
- `colab drivemount -s <name> [PATH]` — mounts Drive at `/content/drive` by default (interactive; not agent-runnable).
- `colab install -s <name> pkg1 pkg2` — installs via `uv pip install --system`, falling back to `pip`. Also `colab install -s <name> -r requirements.txt`.

### Inspect & report
- `colab help` (or `colab help <cmd>`) lists/explains commands; the listing is alphabetical.
- `colab sessions` lists server-side assignments and auto-prunes stale local entries. Orphans with no local record show as `[?]`.
- `colab status [-s <name>]` shows hardware, IDLE/BUSY, and last execution.
- `colab log -s <name> [-n 20] [-t TYPE]` shows recent structured events; invaluable when a task fails.
- `colab log -s <name> -o summary.ipynb` exports the session as a notebook (also `.md`, `.txt`, `.jsonl` by suffix).
- `colab url -s <name>` prints a browser URL that attaches the Colab web UI to your existing CLI session instead of allocating a new VM (add `--open` to launch it).
- `colab skill` / `colab readme` print this skill and the README (handy for self-discovery).

## Safety
- **Always `colab stop -s <name>` when done** — idle VMs burn compute units. `colab run` (without `--keep`) self-cleans even if the script errors.
- Local state lives in `~/.config/colab-cli/sessions.json` (settings in `settings.json`, history in `history/*.jsonl`). Don't edit by hand.
- **Isolate parallel/agent runs** with the global `--config <path>` flag to point session state at a scratch file (e.g. `colab --config /tmp/agent.json new -s job`).

## Recovery
- "Session not found" / 404 / 401 on exec: the backend pruned the VM. `colab exec`/`repl` detect this and clean up local state automatically — run `colab sessions` and re-create with `colab new`.
- Execution timeout or wedged kernel: `colab restart-kernel -s <name>` (keeps the VM, resets the kernel), or `colab stop` then `colab new`.


---

# 운영 노트 — 이 저장소에서 실제로 부딪힌 제약 (2026-10-31 E018 작업)

`colab -h`, `colab help <cmd>`, `colab readme`, `colab skill` 원문을 위에 캡처했다.
아래는 그중 E018 pane 실행에서 실제로 문제가 된 항목만 정리한 것이다.

1. **`colab exec`는 인라인 코드 인자를 받지 않는다.**
   `colab exec -s S "print(1)"` → `Got unexpected extra argument(s)`.
   허용 옵션은 `-s/--session`, `-f/--file`, `--output-image`, `--timeout`, `--env` 뿐이다.
   코드 실행은 `-f script.py` 또는 stdin 파이프(`echo ... | colab exec -s S`)로만 한다.

2. **`-f`가 읽는 파일은 "CLI가 도는 기계"(Mac mini)의 로컬 경로다.**
   VM 안의 `/tmp/x.py`를 가리키면 `FileNotFoundError`가 난다. CLI가 파일 내용을 읽어
   원격 커널로 전송하는 구조이므로, 스크립트는 Mac mini에 두고 `gomac`으로 밀어넣어야 한다.

3. **`colab exec`은 예외/비정상 종료 코드를 전파하지 않는다 — 조용한 실패의 주범.**
   `remote/launch.py`가 `ValueError`로 죽었는데 runner는 성공으로 간주해 poll로 넘어갔고,
   결과는 "No downloadable artifacts"라는 엉뚱한 오류로 나타났다.
   종료 코드가 필요하면 `colab run`(new+exec+stop, exit code 전파, stdout/stderr 분리)을 쓴다.

4. **세션 이름이 stale하면 upload가 실패한다.**
   실패한 실행이 남긴 세션이 `colab status`에 남아 있으면 `colab new`를 건너뛰고 죽은 VM에
   업로드를 시도한다. 재시도 전에 `colab stop -s <name>`으로 반드시 정리한다.

5. **워크플로 기본값**: 기본 cwd는 `/content`, 경로는 절대경로로 쓴다.
   무인 실행에서 `repl`/`console`/`auth`/`drivemount`는 TTY를 요구하므로 쓰지 않는다.
   `colab console`은 VM 안에서 bash를 **tmux로 감싸** 제공한다(= 원격 tmux 세션 수단).

6. **비용 확인**: `colab usage`로 컴퓨트 유닛 잔액을 먼저 본다(E018 시점 46.87 유닛).
   E018 리플레이는 CPU 세션이면 충분하고 1분 내외다. GPU는 실제 학습(E020)에만 쓴다.
