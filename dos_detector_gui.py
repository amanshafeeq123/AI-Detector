import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from dos_ai_module import DosAIDetector
import threading
import queue
import logging
import os

class DosDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DoS Attack Detector")
        self.root.geometry("1200x800")
        
        # creates the main scrollable canvas
        self.main_canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.main_canvas)
        
        # configures the canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # packs the scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", fill="both", expand=True)
        
        # adds the mousewheel scrolling (uhh might need to fix since it's a bit janky but it might be my pc)
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # initializes the detector
        self.detector = DosAIDetector()
        
        # creates the queue for thread communication
        self.queue = queue.Queue()
        
        # creates the GUI elements
        self.create_widgets()
        
        # initializes the plot windows
        self.plot_windows = {}
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def create_widgets(self):
        # main frame
        main_frame = ttk.Frame(self.scrollable_frame, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # file selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # input file selection
        ttk.Label(file_frame, text="Input CSV:").grid(row=0, column=0, sticky=tk.W)
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        
        # output directory selection
        ttk.Label(file_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_path = tk.StringVar(value="output")
        ttk.Entry(file_frame, textvariable=self.output_path, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, pady=5)
        
        # training configuration frame
        train_frame = ttk.LabelFrame(main_frame, text="Training Configuration", padding="5")
        train_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # epoch configuration
        ttk.Label(train_frame, text="Number of Epochs:").grid(row=0, column=0, sticky=tk.W)
        self.epochs = tk.StringVar(value="30")
        ttk.Entry(train_frame, textvariable=self.epochs, width=10).grid(row=0, column=1, padx=5)
        
        # epoch update frequency
        ttk.Label(train_frame, text="Update Every N Epochs:").grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.update_frequency = tk.StringVar(value="5")
        ttk.Entry(train_frame, textvariable=self.update_frequency, width=10).grid(row=0, column=3, padx=5)
        
        # progress bar
        self.progress = ttk.Progressbar(main_frame, length=400, mode='determinate')
        self.progress.grid(row=2, column=0, columnspan=3, pady=10)
        
        # status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=3, column=0, columnspan=3)
        
        # analysis button
        self.analyze_btn = ttk.Button(main_frame, text="Start Analysis", command=self.start_analysis)
        self.analyze_btn.grid(row=4, column=0, columnspan=3, pady=10)
        
        # console frame with scrollbar
        console_frame = ttk.Frame(main_frame)
        console_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # console label
        ttk.Label(console_frame, text="Console Output:").pack(anchor=tk.W)
        
        # console output with scrollbar
        console_scroll = ttk.Scrollbar(console_frame)
        self.console = tk.Text(console_frame, height=10, width=80, yscrollcommand=console_scroll.set)
        console_scroll.config(command=self.console.yview)
        
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        console_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # plot display area
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=6, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # makes the notebook expandable
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        # configures the logging to the console
        self.setup_logging()
    
    def setup_logging(self):
        # creates the custom handler that writes to the console widget
        class ConsoleHandler(logging.Handler):
            def __init__(self, console):
                logging.Handler.__init__(self)
                self.console = console
            
            def emit(self, record):
                msg = self.format(record)
                self.console.insert(tk.END, msg + '\n')
                self.console.see(tk.END)  
                self.console.update_idletasks()  
        
        # adds the handler to the logger
        console_handler = ConsoleHandler(self.console)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(console_handler)
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Wireshark CSV Export",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=os.getcwd()
        )
        if directory:
            self.output_path.set(directory)
    
    def start_analysis(self):
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a CSV file first")
            return
        
        # validates epoch inputs
        try:
            epochs = int(self.epochs.get())
            update_freq = int(self.update_frequency.get())
            if epochs < 1 or update_freq < 1 or update_freq > epochs:
                raise ValueError("Invalid epoch configuration")
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numbers for epochs and update frequency")
            return
        
        # creates output directory if it doesn't exist
        output_dir = self.output_path.get()
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create output directory: {str(e)}")
            return
        
        # disables the button during analysis
        self.analyze_btn.state(['disabled'])
        self.status_var.set("Analysis in progress...")
        self.progress['value'] = 0
        
        # initializes the detector with the output directory and training parameters
        self.detector = DosAIDetector(output_dir=output_dir)
        self.detector.training_config = {
            'epochs': epochs,
            'update_frequency': update_freq
        }
        
        # starts the analysis in a separate thread
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
        
        # starts checking the queue for updates
        self.root.after(100, self.check_queue)
    
    def run_analysis(self):
        try:
            # loads and processes the data
            self.queue.put(('progress', 10))
            df = self.detector.load_csv(self.file_path.get())
            
            self.queue.put(('progress', 30))
            processed_df = self.detector.preprocess_data(df)
            
            # trains the models
            self.queue.put(('progress', 50))
            results_df, history = self.detector.train_models(processed_df)
            
            # generates the visualizations
            self.queue.put(('progress', 70))
            plots = self.detector.generate_visualizations(results_df)
            
            # saves the models
            self.queue.put(('progress', 90))
            self.detector.save_models()
            
            # gets the summary
            summary = self.detector.summarize_results(results_df)
            
            # updates the GUI with the results
            self.queue.put(('plots', plots))
            self.queue.put(('summary', summary))
            self.queue.put(('progress', 100))
            self.queue.put(('status', "Analysis completed successfully!"))
            
        except Exception as e:
            self.queue.put(('error', str(e)))
    
    def check_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                
                if msg_type == 'progress':
                    self.progress['value'] = data
                elif msg_type == 'status':
                    self.status_var.set(data)
                elif msg_type == 'error':
                    messagebox.showerror("Error", data)
                    self.analyze_btn.state(['!disabled'])
                    self.status_var.set("Ready")
                elif msg_type == 'plots':
                    self.display_plots(data)
                elif msg_type == 'summary':
                    self.display_summary(data)
                
                self.queue.task_done()
                
        except queue.Empty:
            if self.progress['value'] != 100:
                self.root.after(100, self.check_queue)
            else:
                self.analyze_btn.state(['!disabled'])
    
    def display_plots(self, plots):
        # clears the existing plot tabs
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        
        # creates new tabs for each plot
        for name, fig in plots.items():
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name.replace('_', ' ').title())
            
            # creates a scrollable canvas for the plot
            canvas_frame = ttk.Frame(frame)
            canvas_frame.pack(fill=tk.BOTH, expand=True)
            
            # if this is the anomaly plot, enhance it
            if 'anomaly' in name.lower():
                # adjusts figure size for better visibility
                fig.set_size_inches(12, 6)
                
                # gets the axes and adjusts the scatter plot
                ax = fig.gca()
                
                # increases marker size and adds alpha for better visibility
                for collection in ax.collections:
                    collection.set_sizes([50])  # larger markers
                    collection.set_alpha(0.6)   # semi-transparent
                
                # adds grid for better readability
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # enhances legend
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # adds the navigation toolbar
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # adds the metrics tab
        self.add_metrics_tab()
    
    def add_metrics_tab(self):
        """Add a tab displaying detailed metrics."""
        metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(metrics_frame, text="Detailed Metrics")
        
        # creates a text widget for metrics
        metrics_text = tk.Text(metrics_frame, wrap=tk.WORD, padx=10, pady=10)
        metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # formats and displays the metrics
        metrics_text.tag_configure("heading", font=("TkDefaultFont", 10, "bold"))
        metrics_text.tag_configure("value", font=("TkDefaultFont", 10))
        
        metrics = self.detector.metrics
        
        # adds model performance section
        metrics_text.insert(tk.END, "Model Performance Metrics\n", "heading")
        metrics_text.insert(tk.END, "\nIsolation Forest:\n", "heading")
        metrics_text.insert(tk.END, f"MSE: {metrics['isolation_forest_mse']:.4f}\n", "value")
        metrics_text.insert(tk.END, f"Accuracy: {metrics['isolation_forest_accuracy']:.2f}%\n", "value")
        
        metrics_text.insert(tk.END, "\nAutoencoder:\n", "heading")
        metrics_text.insert(tk.END, f"MSE: {metrics['autoencoder_mse']:.4f}\n", "value")
        metrics_text.insert(tk.END, f"Accuracy: {metrics['autoencoder_accuracy']:.2f}%\n", "value")
        
        metrics_text.insert(tk.END, "\nCombined Model:\n", "heading")
        metrics_text.insert(tk.END, f"Accuracy: {metrics['combined_accuracy']:.2f}%\n", "value")
        
        # makes the text widget read-only
        metrics_text.configure(state='disabled')
    
    def display_summary(self, summary):
        # creates the summary window
        window = tk.Toplevel(self.root)
        window.title("Analysis Summary")
        window.geometry("600x400")
        
        # adds the summary text
        text = tk.Text(window, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)
        
        # formats and inserts the summary
        text.insert(tk.END, "=== DoS Attack Analysis Summary ===\n\n")
        
        # basic statistics
        text.insert(tk.END, "Basic Statistics:\n")
        text.insert(tk.END, f"Total Packets: {summary['total_packets']}\n")
        text.insert(tk.END, f"Anomalies Detected: {summary['anomalies_detected']}\n")
        text.insert(tk.END, f"Anomaly Percentage: {summary['anomaly_percentage']}\n\n")
        
        # model metrics
        text.insert(tk.END, "Model Performance Metrics:\n")
        metrics = summary['model_metrics']
        text.insert(tk.END, f"Isolation Forest MSE: {metrics['isolation_forest_mse']}\n")
        text.insert(tk.END, f"Autoencoder MSE: {metrics['autoencoder_mse']}\n")
        text.insert(tk.END, f"Isolation Forest Accuracy: {metrics['isolation_forest_accuracy']}\n")
        text.insert(tk.END, f"Autoencoder Accuracy: {metrics['autoencoder_accuracy']}\n")
        text.insert(tk.END, f"Combined Model Accuracy: {metrics['combined_accuracy']}\n\n")
        
        # protocol distribution
        text.insert(tk.END, f"\nProtocol Distribution:\n")
        for protocol, count in summary['protocols'].items():
            text.insert(tk.END, f"  {protocol}: {count}\n")
        
        text.configure(state='disabled')
        
        # adds save button
        save_frame = ttk.Frame(window)
        save_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(save_frame, text="Save Summary", command=lambda: self.save_summary(summary)).pack(side=tk.RIGHT)
    
    def save_summary(self, summary):
        """Save the analysis summary to a file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Analysis Summary"
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("=== DoS Attack Analysis Summary ===\n\n")
                    
                    # writes basic statistics
                    f.write("Basic Statistics:\n")
                    f.write(f"Total Packets: {summary['total_packets']}\n")
                    f.write(f"Anomalies Detected: {summary['anomalies_detected']}\n")
                    f.write(f"Anomaly Percentage: {summary['anomaly_percentage']}\n\n")
                    
                    # writes model metrics
                    f.write("Model Performance Metrics:\n")
                    metrics = summary['model_metrics']
                    f.write(f"Isolation Forest MSE: {metrics['isolation_forest_mse']}\n")
                    f.write(f"Autoencoder MSE: {metrics['autoencoder_mse']}\n")
                    f.write(f"Isolation Forest Accuracy: {metrics['isolation_forest_accuracy']}\n")
                    f.write(f"Autoencoder Accuracy: {metrics['autoencoder_accuracy']}\n")
                    f.write(f"Combined Model Accuracy: {metrics['combined_accuracy']}\n\n")
                    
                    # writes protocol distribution
                    f.write("Protocol Distribution:\n")
                    for protocol, count in summary['protocols'].items():
                        f.write(f"  {protocol}: {count}\n")
                
                messagebox.showinfo("Success", "Summary saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save summary: {str(e)}")

def main():
    root = tk.Tk()
    app = DosDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
