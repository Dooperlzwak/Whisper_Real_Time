import customtkinter as ctk
from app.gui import TranscriptionApp

def main():
    root = ctk.CTk()
    app = TranscriptionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()