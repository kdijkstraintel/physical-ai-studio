from time import sleep
import numpy as np
import time
from typing import Optional
import cv2
from torchcodec.decoders import VideoDecoder

class Pacer:
    fps: float
    speed: float = 1.0   # 1.0 = realtime, 0.5 = half speed, 2.0 = double speed

    def __init__(self, fps: float, speed: float):
        self.fps = fps
        self.speed = speed
        self.period = 1.0 / (self.fps * self.speed)
        self.t0: Optional[float] = None
        self.k: int = -1  # last emitted frame index

    def start(self) -> None:
        self.t0 = time.perf_counter()
        self.k = -1

    def next_sleep(self) -> float:
        """Return seconds to sleep so the NEXT frame lands on schedule."""
        if self.t0 is None:
            self.start()
        self.k += 1
        target = self.t0 + self.k * self.period
        now = time.perf_counter()
        return max(0.0, target - now)

    def frames_behind(self) -> int:
        """How many frame slots have elapsed beyond our next slot (for drop policy)."""
        if self.t0 is None:
            return 0
        now = time.perf_counter()
        target_k = int((now - self.t0) / self.period)
        return max(0, target_k - self.k)

class Capture:
    def __init__(self, source: Optional = None, name: str = None):
        self.source = source
        if name is not None:
            self.name = name
        else:
            self.name = source

    def capture(self):
        raise NotImplementedError()

class CropAndResize(Capture):
    def __init__(self, wrapped_capture: Capture, crop: Optional[tuple[int, int, int, int]] = None, resize: Optional[tuple[int, int]] = None):
        super().__init__()
        self.source = wrapped_capture.source
        self.wrapped_capture = wrapped_capture
        self.crop = crop
        self.resize = resize

    def capture(self):
        frame = self.wrapped_capture.capture()
        if self.crop is not None:
            t, l, h, w = self.crop
            frame = frame[t:t + h, l:l + w, :]

        if self.resize is not None:
            h, w = self.resize
            frame = cv2.resize(frame, (w, h))

        return frame

class OpenCVCapture(Capture):
    def __init__(self, source, resolution: Optional[tuple[int, int]] = None, fps: Optional[int] = None, name: str = None):

        super().__init__(source, name)
        self.resolution = resolution
        self.fps = fps

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera: {self.source}")

        if self.resolution is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[1])
        if self.fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def __del__(self):
        self.cap.release()

    def capture(self):
        _, frame =self.cap.read()
        return frame

class TorchCapture(Capture):
    def __init__(self, source, speed: float= 1.0, name: str = None):
        super().__init__(source, name)

        self.reader = VideoDecoder(self.source, dimension_order="NHWC", device="cpu")
        fps = float(getattr(self.reader.metadata, "average_fps", 0) or 0) or 30.0
        self.read_iterator = iter(self.reader)
        self.pacer = Pacer(fps=fps, speed=speed)
        self.pacer.start()

    def __del__(self):
        self.reader = None

    def capture(self):
        # load all frames until last requested frame
        try:
            # for _ in range(self.pacer.frames_behind()):
            #    next(self.read_iterator)
            frame = next(self.read_iterator)
        except StopIteration:
            self.pacer.start()
            self.read_iterator = iter(self.reader)
            frame = next(self.read_iterator)

        sleep_s = self.pacer.next_sleep()
        if sleep_s > 0:
            sleep(sleep_s)

        return frame.numpy()[:, :, ::-1]

class OverLayCapture(Capture):
    def __init__(self, captures: list[Capture], name: str = None):
        super().__init__(None, name)
        self.captures = captures

    def capture(self):
        factor = 1/len(self.captures)
        image = None
        for capture in self.captures:
            if image is None:
                image = capture.capture() * factor
            else:
                image += capture.capture() * factor
        return image.astype(np.uint8)

class LiveView:
    def __init__(self, captures: list[Capture], window_name: str = "View"):
        self.captures = captures
        self.window_name = window_name

    def display_loop(self) -> None:
        try:
            while True:
                for c in self.captures:
                    image = c.capture()
                    cv2.imshow(f"{self.window_name} {c.name}", image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
