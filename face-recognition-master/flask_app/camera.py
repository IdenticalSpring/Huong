import cv2

class VideoCamera:
    def __init__(self, source=0):
        """
        Initialize the camera.
        Args:
            source (int or str): The video source. Use `0` for the default webcam,
                                or provide an IP camera URL or a video file path.
        """
        self.video = cv2.VideoCapture(source)
        if not self.video.isOpened():
            raise RuntimeError(f"Unable to open video source {source}")

    def __del__(self):
        """
        Release the video resource when the object is destroyed.
        """
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        """
        Capture a single frame from the video source.
        Returns:
            frame (numpy.ndarray): The captured frame in BGR format.
        """
        success, frame = self.video.read()
        if not success:
            return None
        return frame

    def get_encoded_frame(self):
        """
        Capture a single frame and encode it as JPEG for streaming.
        Returns:
            bytes: The JPEG-encoded frame.
        """
        frame = self.get_frame()
        if frame is None:
            return None
        # Encode the frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
