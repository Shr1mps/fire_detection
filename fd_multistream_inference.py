import tkinter as tk
import cv2
from ultralytics import YOLO
from threading import Thread, Event

class StoppableThread(Thread):
    def __init__(self, target, args=()):
        super().__init__(target=target, args=args)
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

def thread_safe_predict(src_path, stop_event):
    model = YOLO(r'model\best.pt')
    results = model.predict(src_path, show=True, conf=0.6, stream=True)
    for r in results:
        if stop_event.is_set():
            print("Stopping thread for:", src_path)
            break
        boxes = r.boxes  # Objeto output de bounding boxes 
        masks = r.masks  # Objeto mascaras de segmentación
        probs = r.probs  # Clase outputs de clasificación

class ThreadManager:
    def __init__(self):
        self.threads = []

    def start_thread(self, target, args):
        thread = StoppableThread(target=target, args=args)
        thread.start()
        self.threads.append(thread)

    def stop_all(self):
        for thread in self.threads:
            thread.stop()
        for thread in self.threads:
            thread.join()

def create_gui(stop_event):
    def on_stop():
        stop_event.set()

    root = tk.Tk()
    root.title("Thread Control")

    stop_button = tk.Button(root, text="Stop Threads", command=on_stop)
    stop_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    source = r"media\fireVideo1.avi"
    source2 = r"media\no_fuego.MOV"
    source3 = r"media\fireVideo2.avi"

    stop_event = Event()

    manager = ThreadManager()
    manager.start_thread(thread_safe_predict, (source, stop_event))
    manager.start_thread(thread_safe_predict, (source2, stop_event))
    manager.start_thread(thread_safe_predict, (source3, stop_event))

    create_gui(stop_event)
    
    manager.stop_all()