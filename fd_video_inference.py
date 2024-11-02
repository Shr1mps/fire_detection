import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    # Carga del modelo
    model = YOLO(r'model\best.pt')


    video_path = PATH
    source2 = SECOND_PATH
    #Apertura del Stream de video
    cap = cv2.VideoCapture(source2)

    # Loop por todos los fotogramas
    while cap.isOpened():
        # Leer un fotograma del video
        success, frame = cap.read()

        if success:
            # Inferencia en un fotograma 
            results = model(frame, conf = 0.6)

            # Dibujar los resultados en el fotograma
            annotated_frame = results[0].plot()
            
            print(results.len())

            # Display del fotograma
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break loop si la tecla 'q' se presiona
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break loop si se llega al final del video
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()