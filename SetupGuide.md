# Setup Guide — Neural Channel Autoencoder

## Team Shannon | AI Era Course Project

This guide walks you through running the program from scratch on **Mac** or **Windows PC**, even if you've never used Python or VS Code before.

---

## PART 1 — Install Python

### Mac

1. Open the **Terminal** app (search for "Terminal" in Spotlight, or find it in Applications → Utilities).
2. Type the following command and press Enter:

```
python3 --version
```

3. If you see a version number like `Python 3.11.5`, Python is already installed — skip to Part 2.
4. If you get an error or nothing happens, install Python:
   - Go to **https://www.python.org/downloads/**
   - Click the big yellow **"Download Python 3.x.x"** button.
   - Open the downloaded `.pkg` file and follow the installer prompts.
   - When finished, close and reopen Terminal, then run `python3 --version` again to confirm.

### Windows PC

1. Open **Command Prompt** (press the Windows key, type `cmd`, and press Enter).
2. Type the following command and press Enter:

```
python --version
```

3. If you see a version number like `Python 3.11.5`, Python is already installed — skip to Part 2.
4. If you get an error or the Microsoft Store opens, install Python manually:
   - Go to **https://www.python.org/downloads/**
   - Click the big yellow **"Download Python 3.x.x"** button.
   - **IMPORTANT:** On the first screen of the installer, check the box that says **"Add Python to PATH"** before clicking "Install Now."
   - When finished, close and reopen Command Prompt, then run `python --version` again to confirm.

---

## PART 2 — Download the Project

1. Download the `shannon_project` folder and save it somewhere easy to find (for example, your Desktop or Documents folder).
2. Make sure the folder structure looks like this:

```
shannon_project/
├── app.py
├── autoencoder_engine.py
├── README.md
├── SETUP_GUIDE.md
├── templates/
│   └── index.html
└── static/
```

---

## PART 3 — Install Required Python Packages

You need to install four libraries that the program depends on. This only needs to be done once.

### Mac

Open Terminal, then copy and paste this entire command and press Enter:

```
pip3 install numpy scipy matplotlib flask
```

If you get a permissions error, try this instead:

```
pip3 install --user numpy scipy matplotlib flask
```

### Windows PC

Open Command Prompt, then copy and paste this entire command and press Enter:

```
pip install numpy scipy matplotlib flask
```

If you get a permissions error, try this instead:

```
pip install --user numpy scipy matplotlib flask
```

### What these packages do

- **numpy** — Math and number-crunching engine (powers the neural network)
- **scipy** — Scientific functions (used for error rate calculations)
- **matplotlib** — Generates all the charts and plots
- **flask** — Runs the web interface in your browser

### Troubleshooting package installation

- **"pip is not recognized":** Python was installed without being added to your system PATH. Reinstall Python and make sure to check "Add Python to PATH" (Windows) or use `python3 -m pip install` instead of `pip install`.
- **"Permission denied":** Add `--user` to the end of the pip command (shown above).
- **"No module named pip":** Run `python3 -m ensurepip --upgrade` (Mac) or `python -m ensurepip --upgrade` (Windows) first, then retry.

---

## PART 4 — Run the Program

### Mac

1. Open Terminal.
2. Navigate to the project folder. If you saved it on your Desktop, type:

```
cd ~/Desktop/shannon_project
```

3. Start the program:

```
python3 app.py
```

4. You should see output like:

```
 * Running on http://0.0.0.0:5000
```

5. Open your web browser (Chrome, Safari, Firefox) and go to:

```
http://localhost:5000
```

The application will load in your browser.

### Windows PC

1. Open Command Prompt.
2. Navigate to the project folder. If you saved it on your Desktop, type:

```
cd %USERPROFILE%\Desktop\shannon_project
```

3. Start the program:

```
python app.py
```

4. You should see output like:

```
 * Running on http://0.0.0.0:5000
```

5. Open your web browser (Chrome, Edge, Firefox) and go to:

```
http://localhost:5000
```

The application will load in your browser.

### Troubleshooting launch

- **"No module named flask":** The packages didn't install correctly. Go back to Part 3 and re-run the pip install command.
- **"Address already in use":** Another program is using port 5000. Either close that program, or edit `app.py` — change the last line to `app.run(debug=False, port=8080)` and then visit `http://localhost:8080` instead.
- **"python: command not found" (Mac):** Use `python3` instead of `python`.
- **Windows Firewall popup:** Click "Allow access" — this only allows connections on your own computer.

---

## PART 5 — Using the Program

Once the application is open in your browser, follow this workflow:

### Step 1: View the Architecture

- Click the **"Architecture"** tab.
- Choose your message bits (k), channel uses (n), and channel type.
- Click **"Show Architecture"** to see a diagram of the neural network.

### Step 2: Train the Model

- Click the **"Train"** tab.
- Set your parameters (the defaults of k=4, n=7, AWGN, 400 epochs work well for a demo).
- Click **"Start Training"** and watch the live progress.
- Training takes 30 seconds to 2 minutes depending on your computer and settings.
- You'll see live metrics (loss, BER, SNR schedule) updating in real time.

### Step 3: View Results

- Click the **"Results"** tab.
- Click **"Generate BER Curve"** to see how the neural autoencoder compares against classical codes (Hamming, Repetition, uncoded BPSK).
- Click **"Show Constellation"** to see the encoding pattern the AI discovered.
- Use the SNR slider and click **"Visualize Channel"** to see how noise corrupts the signal.

### Step 4: Live Demo

- Click the **"Live Demo"** tab.
- Type any message in the text box.
- Adjust the SNR slider (higher = cleaner signal, lower = more noise).
- Click **"Transmit"** to send your message through the noisy channel and see the AI recover it.
- Click **"SNR Sweep"** to test across many noise levels at once.

---

## PART 6 — Stopping the Program

To stop the program, go back to your Terminal (Mac) or Command Prompt (Windows) and press **Ctrl + C** on your keyboard. This shuts down the web server. You can close the browser tab at any time.

---

## Quick Reference

| Action | Mac Command | Windows Command |
|--------|-------------|-----------------|
| Open terminal | Terminal app | `cmd` from Start |
| Check Python version | `python3 --version` | `python --version` |
| Install packages | `pip3 install numpy scipy matplotlib flask` | `pip install numpy scipy matplotlib flask` |
| Navigate to folder | `cd ~/Desktop/shannon_project` | `cd %USERPROFILE%\Desktop\shannon_project` |
| Run the program | `python3 app.py` | `python app.py` |
| Open in browser | `http://localhost:5000` | `http://localhost:5000` |
| Stop the program | Ctrl + C | Ctrl + C |

---

## Optional — Using VS Code

If you prefer using Visual Studio Code:

1. Download VS Code from **https://code.visualstudio.com/**
2. Install it and open it.
3. Go to **File → Open Folder** and select the `shannon_project` folder.
4. Open the built-in terminal: go to **Terminal → New Terminal** (or press Ctrl + ` on the keyboard).
5. In the VS Code terminal, run the pip install and python commands from Parts 3 and 4 above. Everything works the same way.

---

*Team Shannon — AI Era Course Project — 2026*

