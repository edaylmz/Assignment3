import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from audio.recorder import record_audio
from audio.utils import add_noise
from features.mfcc import compute_mfcc
from config import MEL_FILTERS, RECORDINGS_DIR

import simpleaudio as sa
from librosa.display import specshow
from librosa import load, effects


DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
last_recorded_path = None  # Will hold last audio path for replay


def record_and_process(digit_var, mel_var, canvas_frame, noise_var, status_label, replay_btn):
    global last_recorded_path

    digit = digit_var.get()
    mel_count = int(mel_var.get())
    apply_noise = noise_var.get()

    status_label.config(text="Recording...")
    status_label.update()

    # File naming
    existing = [f for f in os.listdir(RECORDINGS_DIR) if f.startswith(digit)]
    count = len(existing) + 1
    filename = f"{digit}_{count}.wav"
    filepath = record_audio(filename)
    last_recorded_path = filepath
    replay_btn.config(state="normal")  # Enable replay button

    status_label.config(text="Processing...")
    status_label.update()

    # Load and optionally add noise
    y, sr = load(filepath, sr=16000)
    y, _ = effects.trim(y)

    if apply_noise:
        y = add_noise(y, noise_level=0.005)

    # Compute features
    log_mel, mfcc = compute_mfcc(filepath, n_mels=mel_count)

    # Export MFCC CSV
    pd.DataFrame(mfcc).to_csv(filepath.replace(".wav", "_mfcc.csv"), index=False)

    # Clear previous plots
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle(f"{digit.capitalize()} #{count} - {mel_count} Mel Filters" + (" (with noise)" if apply_noise else ""))

    im1 = specshow(log_mel, sr=16000, x_axis='time', y_axis='mel', fmax=7000, ax=axs[0])
    axs[0].set_title("Log-Mel Spectrogram")
    fig.colorbar(im1, ax=axs[0], format="%+2.0f dB")

    im2 = specshow(mfcc, sr=16000, x_axis='time', ax=axs[1])
    axs[1].set_title("MFCCs (13 Coefficients)")
    fig.colorbar(im2, ax=axs[1])

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    status_label.config(text="Done!")


def replay_audio():
    global last_recorded_path
    if last_recorded_path and os.path.exists(last_recorded_path):
        wave_obj = sa.WaveObject.from_wave_file(last_recorded_path)
        wave_obj.play()

def launch_interface():
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    root = tk.Tk()
    root.title("Digit Recorder & MFCC Viewer")

    top_frame = tk.Frame(root)
    top_frame.pack(padx=10, pady=10)

    # Digit
    tk.Label(top_frame, text="Digit:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    digit_var = tk.StringVar()
    digit_combo = ttk.Combobox(top_frame, textvariable=digit_var, values=DIGITS, state="readonly", width=15)
    digit_combo.grid(row=0, column=1, padx=5, pady=5)
    digit_combo.current(0)

    # Mel filter
    tk.Label(top_frame, text="Mel Filters:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    mel_var = tk.StringVar(value=str(MEL_FILTERS[0]))
    mel_combo = ttk.Combobox(top_frame, textvariable=mel_var, values=[str(f) for f in MEL_FILTERS], state="readonly", width=15)
    mel_combo.grid(row=1, column=1, padx=5, pady=5)

    # Add noise checkbox
    noise_var = tk.BooleanVar()
    noise_check = tk.Checkbutton(top_frame, text="Apply background noise", variable=noise_var)
    noise_check.grid(row=2, column=0, columnspan=2, pady=5)

    # Status label
    status_label = tk.Label(top_frame, text="Idle", fg="blue")
    status_label.grid(row=3, column=0, columnspan=2, pady=5)

    # Record button
    record_btn = tk.Button(top_frame, text=" Record & Analyze", command=lambda: record_and_process(
        digit_var, mel_var, canvas_frame, noise_var, status_label, replay_btn))
    record_btn.grid(row=4, column=0, columnspan=2, pady=10)

    # Replay button
    replay_btn = tk.Button(top_frame, text="Replay Last Recording", state="disabled", command=replay_audio)
    replay_btn.grid(row=5, column=0, columnspan=2)

    # Canvas for plots
    canvas_frame = tk.Frame(root)
    canvas_frame.pack(padx=10, pady=10, fill="both", expand=True)

    root.mainloop()
