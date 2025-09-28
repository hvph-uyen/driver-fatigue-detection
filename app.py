import csv
import time
import threading
from pathlib import Path
import webview
import pandas as pd
# import serial  # Uncomment nếu dùng Arduino

from model import load_model_and_scaler, predict_from_csv

# ----------------- APP -----------------
class StatusApp:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.current_status = 'SAFE'
        self.prev_label = 'A'
        self.data_buffer = []
        self.running = True

        self.data_source = 'data/C.csv'  # CSV file to read
        
        # self.ser = serial.Serial('COM3', 9600, timeout=1)
        # time.sleep(2)  # chờ Arduino khởi động

        self.window = webview.create_window(
            'Status Monitor',
            url=self.get_file_url('safe'),
            width=500,
            height=350,
            resizable=False
        )

        threading.Thread(target=self.read_csv_loop, daemon=True).start()
        threading.Thread(target=self.monitor_loop, daemon=True).start()

    def get_file_url(self, name):
        base = Path(__file__).resolve().parent
        path = base / 'templates' / f'{name}.html'
        return path.as_uri()

    def read_csv_loop(self):
        df = pd.read_csv(self.data_source)
        for _, row in df.iterrows():
            if not self.running:
                break
            t = row['time']
            swa = row['steering_angle_deg']
            yr = row['yaw_rate_deg']
            self.data_buffer.append([t, swa, yr])
            time.sleep(0.02)  # giả lập 50 Hz
        # self.ser.write(b'A')
        self.running = False  # Dừng monitor loop luôn

    def monitor_loop(self):
        while self.running:
            time.sleep(5)
            if not self.data_buffer:
                continue

            filename = 'data/data.csv'
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'steering_angle_deg', 'yaw_rate_deg'])
                writer.writerows(self.data_buffer)
            print(f"Saved {len(self.data_buffer)} rows to {filename}")

            label = predict_from_csv(filename, self.model, self.scaler)
            print(f"Predicted Label: {label}")

            new_status = self.current_status
            if self.current_status == 'SAFE' and self.prev_label == 'B' and label == 'B':
                new_status = 'DANGER'
            elif self.current_status == 'SAFE' and self.prev_label == 'B' and label == 'C':
                new_status = 'DANGER'
            elif self.current_status == 'SAFE' and self.prev_label == 'C' and label == 'B':
                new_status = 'DANGER'
            elif self.current_status == 'SAFE' and self.prev_label == 'C' and label == 'C':
                new_status = 'SUPERDANGER'
            elif self.current_status == 'DANGER' and self.prev_label == 'C' and label == 'C':
                new_status = 'SUPERDANGER'
                new_url = self.get_file_url(new_status.lower())
                js = f"window.location.href = '{new_url}'"
                self.window.evaluate_js(js)
                # self.ser.write(b'C')
                print("Sent C.")

            if new_status != self.current_status:
                self.current_status = new_status
                print(f"Changing to: {new_status}")

                new_url = self.get_file_url(new_status.lower())
                js = f"window.location.href = '{new_url}'"
                self.window.evaluate_js(js)

                if new_status == 'DANGER':
                    # self.ser.write(b'B')
                    print("Sent B.")
                elif new_status == 'SUPERDANGER':
                    # self.ser.write(b'C')
                    print("Sent C.")
                
            self.prev_label = label
            self.data_buffer.clear()

# ----------------- MAIN -----------------
if __name__ == '__main__':
    model, scaler = load_model_and_scaler()
    app = StatusApp(model, scaler)
    webview.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        app.running = False
