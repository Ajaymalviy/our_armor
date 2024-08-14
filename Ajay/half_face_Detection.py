import cv2
import face_recognition
input_video = cv2.VideoCapture('0')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = input_video.get(cv2.CAP_PROP_FPS)
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# creation of  VideoWriter object
output_video = cv2.VideoWriter("output_video_of_half_video.avi", fourcc, fps, (frame_width, frame_height))

while input_video.isOpened():
    ret, frame = input_video.read()
    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    # Loop through each face found in the frame
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    output_video.write(frame)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the input video and output video objects
input_video.release()
output_video.release()
cv2.destroyAllWindows()
